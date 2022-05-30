import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers_f.src.transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2FeatureProjection,
                                                                               Wav2Vec2Encoder, Wav2Vec2EncoderLayer,
                                                                               Wav2Vec2EncoderStableLayerNorm,
                                                                               Wav2Vec2PreTrainedModel, 
                                                                               Wav2Vec2Model, 
                                                                               Wav2Vec2Config, 
                                                                               Wav2Vec2FeatureExtractor, 
                                                                               Wav2Vec2PositionalConvEmbedding)
from utils import buffered_arange, sample_negatives, ConstrativeLoss
from models import NegativeSampler, TransposeLast, SamePad, SegmentsRepr
from transformers_f.src.transformers.deepspeed import is_deepspeed_zero3_enabled
from typing import Optional, Tuple, Union
import gc
from models import ConvFeatureEncoder

# Приведенные функции основаны на https://github.com/NVIDIA/NeMo и https://github.com/huggingface    

# class NegativeSampler(nn.Module):
    
#     def __init__(self, n_negatives = 1, loss_args = {'reduction': 'mean'}):
#         super(NegativeSampler, self).__init__()
#         self.n_negatives = n_negatives
#         self.loss = ConstrativeLoss(**loss_args)
        
#     def forward(self, x, transpose = False, attention_mask = None):
# #         if transpose:
# #             x = x.transpose(1, 2)
#         targets = torch.roll(x, -1, dims=1)
        
#         negatives, negs_ids = sample_negatives(targets, n_negatives = self.n_negatives, attention_mask = attention_mask)
#         return x, targets, negatives
        
#     def compute_all(self, batch, transpose = False, attention_mask = None):
        
#         x, targets, negatives = self.forward(batch, transpose, attention_mask = attention_mask)
#         loss, acc_score = self.loss(x, targets, negatives)
#         return loss, dict(loss=loss.item(), 
#                           acc=acc_score)

    
# class TransposeLast(torch.nn.Module):
#     """
#     Transposes last dimension. Useful for adding to a sequential block.
#     """

#     def forward(self, x):
#         return x.transpose(-2, -1)


# class SamePad(torch.nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.remove = kernel_size % 2 == 0

#     def forward(self, x):
#         if self.remove:
#             x = x[:, :, :-1]
#         return x
    
# Данная функция основана на https://github.com/felixkreuk/UnsupSeg/blob/master/utils.py
class RMetrics(nn.Module):
    def __init__(self, eps = 1e-5, tolerance = 1, sampling_rate = 16000):
        super(RMetrics, self).__init__()
        self.tolerance = tolerance
        self.eps = eps
        self.sampling_rate = sampling_rate
    
    def calculate_stride(self, isz, conv_layers):
        pad = 0
        insize = isz
        totstride = 1
        sec_per_frame = 1/self.sampling_rate

        for layer in conv_layers:
            kernel, stride = layer
            outsize = (insize + 2*pad - 1*(kernel-1)-1) / stride + 1
            insize = outsize
            totstride = totstride * stride

        RFsize = isz - (outsize - 1) * totstride

        ms_per_frame = sec_per_frame*RFsize*1000
        ms_stride = sec_per_frame*totstride*1000
        return outsize, totstride, RFsize, ms_per_frame, ms_stride
        
    def get_frames(self, secs, stride):
        frames = [[int(i*self.sampling_rate/stride) for i in sec] for sec in secs]
        return frames
        
    def make_true_boundaries(self, secs, boundaries, stride):
        frames = self.get_frames(secs, stride)
        true_boundaries = torch.zeros(size = boundaries.shape)
        for num_frame, frame in enumerate(frames):
            for i in frame:
                true_boundaries[num_frame, i] = 1
        return true_boundaries.long().detach().numpy()
    
    def get_sec_bounds(self, b, stride, attention_mask = None):
        if type(b)==torch.Tensor:
            b1 = b.long().detach().cpu().numpy()
        else:
            b1 = b
        
        if attention_mask is not None:
            b1 = b1*attention_mask.long().detach().cpu().numpy()
            
        frames_pred = []
        secs_pred = []
        for i in range(b1.shape[0]):
            frames = np.where(b1[i, :] == 1)[0]
            secs = [i*stride/self.sampling_rate for i in frames]
            frames_pred.append(frames)
            secs_pred.append(secs)
        return frames_pred, secs_pred
    
    def get_precision_recall_frames(self, true_boundaries, b, attention_mask = None):
        if type(b)==torch.Tensor:
            b1 = b.long().detach().cpu().numpy()
        else:
            b1 = b
            
        if attention_mask is not None:
            b1 = b1*attention_mask.long().detach().cpu().numpy()
            
        recall = recall_score(true_boundaries.flatten(), b1.flatten())
        pre = precision_score(true_boundaries.flatten(), b1.flatten())
        f_score = f1_score(true_boundaries.flatten(), b1.flatten())
        return recall, pre, f_score
    
    def get_stats(self, frames_true, frames_pred):
        
        # Утащено отсюда: https://github.com/felixkreuk/UnsupSeg/blob/68c2c7b9bd49f3fb8f51c5c2f4d5aa85f251eaa8/utils.py#L69
        precision_counter = 0 
        recall_counter = 0
        pred_counter = 0 
        gt_counter = 0

        for (y, yhat) in zip(frames_true, frames_pred):
            for yhat_i in yhat:
                min_dist = np.abs(np.array(y) - yhat_i).min()
                precision_counter += (min_dist <= self.tolerance)
            for y_i in y:
                if len(yhat) > 0:
                    min_dist = np.abs(np.array(yhat) - y_i).min()
                    recall_counter += (min_dist <= self.tolerance)
                else:
                    recall_counter += 0
            pred_counter += len(yhat)
            gt_counter += len(y)

        return precision_counter, recall_counter, pred_counter, gt_counter
    
    def calc_metr(self, precision_counter, recall_counter, pred_counter, gt_counter):

        # Утащено отсюда: https://github.com/felixkreuk/UnsupSeg/blob/68c2c7b9bd49f3fb8f51c5c2f4d5aa85f251eaa8/utils.py#L69
        EPS = 1e-7

        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)

        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval
    
    def get_metrics(self, true_secs, b, seq_len, conv_layers, attention_mask = None, return_secs = False):
        
        outsize, totstride, RFsize, ms_per_frame, ms_stride = self.calculate_stride(seq_len, conv_layers)
#         print(seq_len, outsize, totstride, RFsize, ms_per_frame, ms_stride)
        frames_true = self.get_frames(true_secs, totstride)
        
        frames_pred, secs_pred = self.get_sec_bounds(b, totstride, attention_mask)
        precision_counter, recall_counter, pred_counter, gt_counter = self.get_stats(frames_true, frames_pred)
        precision, recall, f1, rval = self.calc_metr(precision_counter, recall_counter, pred_counter, gt_counter)
        if return_secs:
            return precision, recall, f1, rval, secs_pred
        else:
            return precision, recall, f1, rval

    

# class SegmentsRepr(nn.Module):
#     def __init__(self, thres = 0.05):
#         super(SegmentsRepr, self).__init__()
#         self.thres = thres
    
#     def boundary(self, frames):

#         #batch_size x seq_len x dim
# #         frames_1_plus = frames.roll(1, dims = 1)
# #         cos = torch.cosine_similarity(frames, frames_1_plus, dim=-1)[:, 1:]
# #         cos = torch.cat(cos[:, 0], cos, dim = 1)
        
#         frames_1_plus = frames.roll(1, dims = 1)
#         cos = torch.cosine_similarity(frames, frames_1_plus, dim=-1)[:, 1:]
#         cos = torch.cat([cos[:, 0].unsqueeze(1), cos], dim = 1)
# #         cos[:, -1] = 0
        
#         min_cos = torch.min(cos, dim=1).values
#         min_cos = min_cos.unsqueeze(1).expand(min_cos.shape[0], frames.shape[1])
#         max_cos = torch.max(cos, dim=1).values
#         max_cos = max_cos.unsqueeze(1).expand(max_cos.shape[0], frames.shape[1])
        
#         d = torch.ones(frames.shape[0], (frames.shape[1])).to(frames.device)-(cos - min_cos)/(max_cos-min_cos)

#         d_1_plus = d.roll(-1, dims = 1)
#         d_2_plus = d.roll(-2, dims = 1)
#         d_1_minus = d.roll(1, dims = 1)
#         d_2_minus = d.roll(2, dims = 1)

#         zeros_d = torch.zeros(d.shape).to(frames.device)

#         p_1 = torch.min(torch.max((d-d_1_plus), zeros_d), torch.max((d-d_1_minus), zeros_d))
#         p_1[:, [0, -1]] = 0
#         p_2 = torch.min(torch.max((d-d_2_plus), zeros_d), torch.max((d-d_2_minus), zeros_d))
#         p_2[:, [0, 1]] = 0
#         p_2[:, [-2, -1]] = 0

#         pt = torch.min(torch.max(torch.max(p_1, p_2)-torch.full(p_1.shape, self.thres).to(frames.device), zeros_d), p_1)

#         b_soft = torch.tanh(10*pt)
#         b_hard = torch.tanh(10000000*pt)
#         b_hard1 = (b_hard-b_soft).detach()
#         b = b_soft + b_hard1
        
# #         indexes_o = torch.ones(frames.shape[0], 1).to(frames.device)
#         return b
    
#     def receive_mask(self, b):
#         b_cumsum = torch.cumsum(b, axis=1)
#         num_fr = torch.sum(b, axis=1)
        
#         # Создать arange
#         cc = torch.arange(1, int(torch.max(num_fr))+1).unsqueeze(0).unsqueeze(0).expand(b_cumsum.shape[0], b_cumsum.shape[1], 
#                                                                                        int(torch.max(num_fr))).to(b.device)
#         # Получение индексов
#         b_cumsum1 = b_cumsum.unsqueeze(-1).expand(b_cumsum.shape[0], b_cumsum.shape[1], cc.shape[-1])
#         unres = torch.tanh(10*torch.abs(cc - b_cumsum1))
#         res = torch.ones(unres.shape).to(b.device)-unres
        
#         # Нормирование
#         sums = torch.sum(res, dim=1).unsqueeze(1).expand(res.shape)+unres # + unres чтобы не было null значений на нулях
#         masks = res/sums
#         masks = masks.transpose(2, 1)
        
#         return masks
    
#     def receive_attentions(self, mask, attention_mask):
#         mask1 = mask.transpose(2, 1)
#         att1 = attention_mask.float().unsqueeze(-1).expand(tuple(mask1.shape))
#         res1 = (mask1.to(att1.device)*att1).sum(dim=1)
#         res1[res1>0] = 1
#         return res1
    
#     def forward(self, frames, attention_mask=None, conv_layers = None):
#         # Получение сегментов
#         b = self.boundary(frames)
#         mask = self.receive_mask(b)
        
#         segments = torch.bmm(mask.to(frames.device), frames)
    
#         if attention_mask is not None and conv_layers is not None:
#             attention_mask = self.receive_attentions(mask, attention_mask)
#             return b, mask, segments, attention_mask.long()
#         else:
#             return b, mask, segments




class Wav2Vec2ModelForSegmentation(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
#         self.feature_extractor = ConvFeatureEncoder(conv_layers = [[1, 256, 10, 5], 
#                                                                    [256, 256, 8, 4], 
#                                                                    [256, 256, 4, 2], 
#                                                                    [256, 256, 4, 2], 
#                                                                    [256, 256, 4, 2]],
#                                                     conv_bias = False)
#         self.feature_extractor = ConvFeatureEncoder(conv_layers = [[1, 512, 10, 5], 
#                                                                    [512, 512, 8, 4], 
#                                                                    [512, 512, 4, 2], 
#                                                                    [512, 512, 4, 2], 
#                                                                    [512, 512, 4, 2]],
#                                                     conv_bias = False)
        
#         self.project_extracted = nn.Linear(256, 64)
        
        self.feature_projection = Wav2Vec2FeatureProjection(config)

#         self.project_back = nn.Linear(256, 512)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None
        
        self.negative_sampler = NegativeSampler(n_negatives = 1, loss_args = {'reduction': 'mean'})
        self.segment_mean = SegmentsRepr(thres = 0.05)
        self.metr = RMetrics()
        self.num_epoch = 2
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values,
        attention_mask=None,
        return_secs = False
    ):
        
        conv_layers_list = [(kernel_size, stride) for kernel_size, stride in zip(self.config.conv_kernel, 
                                                                                 self.config.conv_stride)]
        seq_len = input_values.shape[1]
        
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
#         extract_features = self.project_extracted(extract_features)
        
        attention_mask1 = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )
        
        extract_features, targets_frames, negatives_frames = self.negative_sampler(extract_features,
                                                                                   attention_mask=attention_mask1)
        
        # compute reduced attention_mask corresponding to feature vectors
        
        frame_loss, frame_acc_score = self.negative_sampler.loss(extract_features, targets_frames, negatives_frames, 
                                                                attention_mask=attention_mask1)
        b, mask, x, attention_mask = self.segment_mean(extract_features, attention_mask1, 
                                                       conv_layers = conv_layers_list)
        
        if return_secs:
            outsize, totstride, RFsize, ms_per_frame, ms_stride = self.segment_mean.calculate_stride(seq_len,
                                                                                                     conv_layers_list)
            frames_pred, secs_pred = self.segment_mean.get_sec_bounds(b, totstride, attention_mask)
            
#         x = self.project_back(x)
        hidden_states, extract_features = self.feature_projection(x)

        x, targets_segments, negatives_segments = self.negative_sampler(hidden_states, attention_mask=attention_mask)
        
        attention_mask = attention_mask.bool()
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        segment_loss, segment_acc_score = self.negative_sampler.loss(hidden_states, targets_segments,
                                                                     negatives_segments, 
                                                                     attention_mask=attention_mask)
        
        if num_epoch >= self.num_epoch:
            loss = frame_loss+segment_loss
        else:
            loss = frame_loss
        
        gc.collect()
        
        if return_secs:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              segment_loss = segment_loss.item(), 
                              segment_acc_score = segment_acc_score, 
                              loss = loss.item(),
                              secs_pred = secs_pred,
                              hidden_states = hidden_states)
        else:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              segment_loss = segment_loss.item(), 
                              segment_acc_score = segment_acc_score, 
                              loss = loss.item(),
                              hidden_states = hidden_states)


    def compute_all(self, input_values, boundaries, num_epoch, attention_mask = None, return_secs=False):
        
        conv_layers_list = [(kernel_size, stride) for kernel_size, stride in zip(self.config.conv_kernel, 
                                                                                 self.config.conv_stride)]
#         conv_layers_list = [[i[2], i[3]] for i in self.feature_extractor.conv_layers_list]
        
        seq_len = input_values.shape[1]

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
#         extract_features = self.project_extracted(extract_features)
        
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask1 = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )

        extract_features, targets_frames, negatives_frames = self.negative_sampler(extract_features, 
                                                                                   attention_mask=attention_mask1)
        

        frame_loss, frame_acc_score = self.negative_sampler.loss(extract_features, targets_frames, negatives_frames,
                                                                 attention_mask=attention_mask1)
        
        b, mask, x, attention_mask = self.segment_mean(extract_features, attention_mask1, 
                                                       conv_layers = conv_layers_list)
        if return_secs:
            precision, recall, f1, r_metr, secs_pred = self.metr.get_metrics(boundaries, b, 
                                                                             seq_len, conv_layers_list,
                                                                             attention_mask=attention_mask1, 
                                                                             return_secs = return_secs)
        else:
            precision, recall, f1, r_metr = self.metr.get_metrics(boundaries, b, seq_len, conv_layers_list,
                                                                  attention_mask=attention_mask1, 
                                                                  return_secs = return_secs)
        
#         x = self.project_back(x)
        hidden_states, extract_features = self.feature_projection(x)

        x, targets_segments, negatives_segments = self.negative_sampler(hidden_states, attention_mask=attention_mask)
    
        attention_mask = attention_mask.bool()
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        segment_loss, segment_acc_score = self.negative_sampler.loss(hidden_states, targets_segments, 
                                                                     negatives_segments, 
                                                                     attention_mask=attention_mask)
        
        if num_epoch >= self.num_epoch:
            loss = frame_loss+segment_loss
        else:
            loss = frame_loss
        
        gc.collect()
        
        if return_secs:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              segment_loss = segment_loss.item(), 
                              segment_acc_score = segment_acc_score, 
                              loss = loss.item(), 
                              precision = precision, 
                              recall = recall,
                              f1 = f1,
                              r_metr = r_metr,
                              secs_pred = secs_pred)
        else:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              segment_loss = segment_loss.item(), 
                              segment_acc_score = segment_acc_score, 
                              loss = loss.item(), 
                              precision = precision, 
                              recall = recall,
                              f1 = f1,
                              r_metr = r_metr)
