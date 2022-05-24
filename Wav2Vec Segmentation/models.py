import torch
from torch import nn
from utils import ConstrativeLoss, sample_negatives
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import optim
from collections import defaultdict
from scipy.signal import find_peaks
import torch.nn.functional as F

class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x
    
# Данная функция основана на https://github.com/felixkreuk/UnsupSeg/blob/master/utils.py
class RMetrics(nn.Module):
    def __init__(self, eps = 1e-5, tolerance = 2, sampling_rate = 16000):
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

class ConvFeatureEncoder(nn.Module):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with LeakyReLU activation.
    """

    def __init__(
        self,
        conv_layers = [[1, 256, 10, 5], [256, 256, 8, 4], [256, 256, 4, 2], [256, 256, 4, 2], [256, 256, 4, 2]],
        conv_bias = False
        
        # Mode for feature extractor. default has a single group norm with d groups in the first conv block,
        # whereas layer_norm has layer norms in every block.
        
#         The function fenc was implemented using a deep convolutional neural network. It contains five layers of 1-D strided
#         convolution with kernel sizes of (10,8,4,4,4) and strides of
#         (5,4,2,2,2) and 256 channels per layer. The total downsampling
#         rate of the network is 160. Each convolution layer is followed
#         by Batch-Normalization and Leaky ReLU non-linear activation
#         function. A fully connected layer linearly projects the outputs
#         in a smaller dimension, 64. Overall, the fenc architecture is
#         same as [23]. Similar to [23], SCPC does not utilize a context
#         network at the frame level.
    ):
        super().__init__()

        def block(
            n_in, n_out, k, stride, conv_bias = False, done = False
        ):
            def make_conv(n_in, n_out, k, stride=stride, bias=False):
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if done:
                return nn.Sequential(
                        make_conv(n_in, n_out, k, stride=stride, bias=conv_bias)
                    )
            else:
                return nn.Sequential(
                        make_conv(n_in, n_out, k, stride=stride, bias=conv_bias),
                        nn.BatchNorm1d(n_out),
                        nn.LeakyReLU()
                    )
        
        self.conv_layers_list = conv_layers
        self.conv_layers = nn.ModuleList()
        done = False
        for i, cl in enumerate(conv_layers):
            (in_d, out_d, k, stride) = cl
            if i == len(conv_layers)-1:
                done = True

            self.conv_layers.append(
                block(
                    in_d,
                    out_d,
                    k,
                    stride, 
                    conv_bias = conv_bias,
                    done = done
                )
            )

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    
class AttentionCalc(nn.Module):
    
    def __init__(self):
        super(AttentionCalc, self).__init__()
        pass
    
    def _get_feat_extract_output_lengths(self, input_lengths, conv_layers):
        """
        Computes the output length of the convolutional layers
        Считает длину выхода слоев сверток
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for layer in conv_layers:
            kernel_size, stride = layer
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def get_feature_vector_attention_mask( # Считает attention masks для разреженных conv слоем векторов
        self, feature_vector_length: int, attention_mask: torch.LongTensor, conv_layers
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1] #суммируем незападенные внимания получаем длины

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, conv_layers)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        ) # делаем нулевой вектор
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool() # разворачивает по последнему измерению
        return attention_mask
    
class SegmentsRepr(nn.Module):
    def __init__(self, thres = 0.05):
        super(SegmentsRepr, self).__init__()
        self.thres = thres
    
    def boundary(self, frames):

        #batch_size x seq_len x dim
#         frames_1_plus = frames.roll(1, dims = 1)
#         cos = torch.cosine_similarity(frames, frames_1_plus, dim=-1)[:, 1:]
#         cos = torch.cat(cos[:, 0], cos, dim = 1)
        
        frames_1_plus = frames.roll(1, dims = 1)
        cos = torch.cosine_similarity(frames, frames_1_plus, dim=-1)[:, 1:]
        cos = torch.cat([cos[:, 0].unsqueeze(1), cos], dim = 1)
#         cos[:, -1] = 0
        
        min_cos = torch.min(cos, dim=1).values
        min_cos = min_cos.unsqueeze(1).expand(min_cos.shape[0], frames.shape[1])
        max_cos = torch.max(cos, dim=1).values
        max_cos = max_cos.unsqueeze(1).expand(max_cos.shape[0], frames.shape[1])
        
        d = torch.ones(frames.shape[0], (frames.shape[1])).to(frames.device)-(cos - min_cos)/(max_cos-min_cos)

        d_1_plus = d.roll(-1, dims = 1)
        d_2_plus = d.roll(-2, dims = 1)
        d_1_minus = d.roll(1, dims = 1)
        d_2_minus = d.roll(2, dims = 1)

        zeros_d = torch.zeros(d.shape).to(frames.device)

        p_1 = torch.min(torch.max((d-d_1_plus), zeros_d), torch.max((d-d_1_minus), zeros_d))
        p_1[:, [0, -1]] = 0
        p_2 = torch.min(torch.max((d-d_2_plus), zeros_d), torch.max((d-d_2_minus), zeros_d))
        p_2[:, [0, 1]] = 0
        p_2[:, [-2, -1]] = 0

        pt = torch.min(torch.max(torch.max(p_1, p_2)-torch.full(p_1.shape, self.thres).to(frames.device), zeros_d), p_1)

        b_soft = torch.tanh(10*pt)
        b_hard = torch.tanh(10000000*pt)
        b_hard1 = (b_hard-b_soft).detach()
        b = b_soft + b_hard1
        
        indexes_o = torch.ones(frames.shape[0], 1).to(frames.device)
        return b
    
    def receive_mask(self, b):
        b_cumsum = torch.cumsum(b, axis=1)
        num_fr = torch.sum(b, axis=1)
        
        # Создать arange
        cc = torch.arange(1, int(torch.max(num_fr))+1).unsqueeze(0).unsqueeze(0).expand(b_cumsum.shape[0], b_cumsum.shape[1], 
                                                                                       int(torch.max(num_fr))).to(b.device)
        
        # Получение индексов
        b_cumsum1 = b_cumsum.unsqueeze(-1).expand(b_cumsum.shape[0], b_cumsum.shape[1], cc.shape[-1])
        unres = torch.tanh(10*torch.abs(cc - b_cumsum1))
        res = torch.ones(unres.shape).to(b.device)-unres
        
        # Нормирование
        sums = torch.sum(res, dim=1).unsqueeze(1).expand(res.shape)+unres # + unres чтобы не было null значений на нулях
        masks = res/sums
        masks = masks.transpose(2, 1)
        
        return masks
    
    def receive_attentions(self, mask, attention_mask):
        mask1 = mask.transpose(2, 1)
        att1 = attention_mask.float().unsqueeze(-1).expand(tuple(mask1.shape))
        res1 = (mask1.to(att1.device)*att1).sum(dim=1)
        res1[res1>0] = 1
        return res1
    
    def forward(self, frames, attention_mask=None, conv_layers = None):
        # Получение сегментов
        b = self.boundary(frames)
        mask = self.receive_mask(b)
        
        segments = torch.bmm(mask.to(frames.device), frames)
    
        if attention_mask is not None and conv_layers is not None:
            attention_mask = self.receive_attentions(mask, attention_mask)
            return b, mask, segments, attention_mask.long()
        else:
            return b, mask, segments


class SegmentsEncoder(nn.Module):
    
    def __init__(self, num_feats = 256):
        super(SegmentsEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(num_feats, num_feats),
                                 nn.LeakyReLU(),
                                 nn.Linear(num_feats, num_feats),
                                 nn.LeakyReLU()
                                   )
        
    def forward(self, x):
        # batch_size, seq_len, dim
        return self.enc(x)
    
    
class NegativeSampler(nn.Module):
    
    def __init__(self, n_negatives = 1, loss_args = {'reduction': 'mean'}):
        super(NegativeSampler, self).__init__()
        self.n_negatives = n_negatives
        self.loss = ConstrativeLoss(**loss_args)
        
    def forward(self, x, transpose = False, attention_mask = None):
#         if transpose:
#             x = x.transpose(1, 2)
        targets = torch.roll(x, -1, dims=1)
        
        negatives, negs_ids = sample_negatives(targets, n_negatives = self.n_negatives, attention_mask = attention_mask)
        return x, targets, negatives
        
    def compute_all(self, batch, transpose = False, attention_mask = None):
        
        x, targets, negatives = self.forward(batch, transpose, attention_mask = attention_mask)
        loss, acc_score = self.loss(x, targets, negatives)
        return loss, dict(loss=loss.item(), 
                          acc=acc_score)


class SegmentPredictor(LightningModule):
    
    def __init__(self, input_size = 256, hidden_size = 64, output_size = 256):
        super(SegmentPredictor, self).__init__()
        self.rnn = nn.GRU(input_size = input_size, 
                          hidden_size = hidden_size, 
                          batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
#         print('GRU', x.shape)
        x, _ = self.rnn(x)
#         print('GRU end', x.shape)
        x = self.activation(self.linear(x))
#         print('Linear', x.shape)
        return x


class FinModel(LightningModule):
    
    def __init__(self, cfg 
#                  conv_args = {}, mask_args = {}, segm_enc_args = {}, 
#                  segm_predictor_args = {}, 
#                  loss_args = {"n_negatives": 1, 
#                               "loss_args": {"reduction": "mean"}}, num_epoch = 2
                ):
        super(FinModel, self).__init__()
        self.cfg = cfg
        self.conv_encoder = ConvFeatureEncoder(**cfg.conv_args)
#         self.projection = nn.Linear(256, 64)
        self.frame_predictor = NegativeSampler(**cfg.loss_args)
        self.conv_layers_list = [[i[2], i[3]] for i in self.conv_encoder.conv_layers_list]
        self.segment_mean = SegmentsRepr(**cfg.mask_args)
        self.attention_calc = AttentionCalc()
        self.segment_encoder = SegmentsEncoder(**cfg.segm_enc_args)
        self.segment_recurrent = SegmentPredictor(**cfg.segm_predictor_args)
        self.segment_predictor = NegativeSampler(**cfg.loss_args)
        self.num_epoch = cfg.num_epoch
        self.metr = RMetrics()
        
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.cfg.optimizer == "sgd":
            self.opt = optim.SGD(parameters, lr=self.cfg.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif self.cfg.optimizer == "adam":
            self.opt = optim.Adam(parameters, lr=self.cfg.learning_rate, weight_decay=5e-4)
        elif self.cfg.optimizer == "ranger":
            self.opt = optim_extra.Ranger(parameters, lr=self.cfg.learning_rate, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), 
                                          eps=1e-5,
                                          weight_decay=0)
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.opt}")
        self.scheduler = optim.lr_scheduler.StepLR(self.opt,
                                                   step_size=self.cfg.lr_anneal_step,
                                                   gamma=self.cfg.lr_anneal_gamma)
        
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
#                                                       patience=1, threshold=0.0001, threshold_mode='rel',
#                                                       cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        return [self.opt]    
        
    def forward(self, x, attention_mask):
        x = self.conv_encoder(x)
        x = x.transpose(1, 2)
        
        attention_mask1 = self.attention_calc.get_feature_vector_attention_mask(x.shape[1], attention_mask, 
                                                                                conv_layers=self.conv_layers_list)
        
        b, mask, x, attention_mask = self.segment_mean(x, attention_mask1, conv_layers = self.conv_layers_list)
        if return_secs:
            outsize, totstride, RFsize, ms_per_frame, ms_stride = self.segment_mean.calculate_stride(seq_len,
                                                                                                conv_layers_list)
            frames_pred, secs_pred = self.segment_mean.get_sec_bounds(b, totstride, attention_mask)
        
        x = self.segment_encoder(x)
        x = self.segment_recurrent(x)
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
    
    def compute_all(self, x, secs, num_epoch, attention_mask,
                    return_secs = False):
        
        seq_len = x.shape[1]
        x = self.conv_encoder(x)
        x = x.transpose(1, 2)
        attention_mask1 = self.attention_calc.get_feature_vector_attention_mask(x.shape[1], attention_mask, 
                                                                                conv_layers=self.conv_layers_list)
        x, targets_frames, negatives_frames = self.frame_predictor(x, transpose=False, attention_mask=attention_mask1)
#         print(negatives_frames[0, 0, 0, :] == negatives_frames[1, 0, 0, :])
        
        frame_loss, frame_acc_score = self.frame_predictor.loss(x, targets_frames, negatives_frames, attention_mask=attention_mask1)
        
        b, mask, x, attention_mask = self.segment_mean(x, attention_mask1, conv_layers = self.conv_layers_list)
        if return_secs:
            precision, recall, f1, r_metr, secs_pred = self.metr.get_metrics(secs, b, 
                                                                             seq_len, self.conv_layers_list,
                                                                             attention_mask=attention_mask1, 
                                                                             return_secs = return_secs)
        else:
            precision, recall, f1, r_metr = self.metr.get_metrics(secs, b, seq_len, self.conv_layers_list,
                                                                  attention_mask=attention_mask1, 
                                                                  return_secs = return_secs)
        
        precision, recall, f1, r_metr = self.metr.get_metrics(secs, b, seq_len, self.conv_layers_list, 
                                                              attention_mask=attention_mask1)
        x = self.segment_encoder(x)
        
        x, targets_segments, negatives_segments = self.segment_predictor(x, attention_mask=attention_mask)
#         print(negatives_segments[0, 0, 0, :] == negatives_segments[1, 0, 0, :])
        x = self.segment_recurrent(x)
        segment_loss, segment_acc_score = self.segment_predictor.loss(x, targets_segments, negatives_segments, attention_mask=attention_mask)
        
        if num_epoch >= self.num_epoch:
            loss = frame_loss+segment_loss
        else:
            loss = frame_loss
        
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

    def training_step(self, batch, batch_idx):

        loss, res_dict = self.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device))
        self.log("train_loss", loss)
        for key, value in res_dict.items():
            self.log(key, value)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, res_dict = self.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device))
        self.log("val_loss", loss)
        for key, value in res_dict.items():
            self.log('val_'+key, value)
        return loss

# Данная функция основана на https://github.com/felixkreuk/UnsupSeg
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
# Данная функция основана на https://github.com/felixkreuk/UnsupSeg
class UnsupLoss(nn.Module):
    def __init__(self):
        super(UnsupLoss, self).__init__()

        self.training = True
        self.pred_steps = [1]
    
    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1)
    
    def get_preds(self, z):

        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            pos_pred = self.score(z[:, :-t], z[:, t:])  # score for positive frame
            preds[t].append(pos_pred)

            for _ in range(1):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                else:
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])

                neg_pred = self.score(z[:, :-t], z[batch_reorder][: , time_reorder])  # score for negative random frame
                preds[t].append(neg_pred)
        return preds
    
    def length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask
    
    def loss(self, preds, lengths):
        loss = 0
        lengths = lengths.int()
        for t, t_preds in preds.items():
            mask = self.length_to_mask(lengths - t)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[...,0] * mask
            loss += -out.mean()
        return loss

# Данная функция основана на https://github.com/felixkreuk/UnsupSeg
class GetBound(nn.Module):
    def __init__(self):
        super(GetBound, self).__init__()
    
    def max_min_norm(self, x):
        x -= x.min(-1, keepdim=True)[0]
        x /= x.max(-1, keepdim=True)[0]
        return x

    def replicate_first_k_frames(self, x, k, dim):
        return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)

    def replicate_last_k_frames(self, x, k, dim):
        return torch.cat([x, x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device))], dim=dim)
    
    def detect_peaks(self, x, lengths, prominence=0.1, width=None, distance=None):
        """detect peaks of next_frame_classifier

        Arguments:
            x {Tensor} -- batch of confidence per time
        """ 
        out = []
        for xi, li in zip(x, lengths):
            if type(xi) == torch.Tensor:
                xi = xi.cpu().detach().numpy()
            xi = xi[:li]  # shorten to actual length
            xmin, xmax = xi.min(), xi.max()
            xi = (xi - xmin) / (xmax - xmin)
#             xi = 1 - (xi - xmin) / (xmax - xmin)
            peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)
            if len(peaks) == 0:
                peaks = np.array([len(xi)-1])
            out.append(peaks)
        return out

    def get_boundaries(self, preds):
        preds1 = preds  # get scores of positive pairs
        preds_s = []

        for b in range(preds1.shape[0]):
            preds = preds1[b, :].unsqueeze(0)
            preds = self.replicate_first_k_frames(preds, k=1, dim=1)  # padding
            preds = 1 - self.max_min_norm(preds)  # normalize scores (good for visualizations)
            preds = self.detect_peaks(x=preds,
                                 lengths=[preds.shape[1]]
                                )  # run peak detection on scores
            preds = preds[0] * 160 / 16000
            preds_s.append(preds)
        return preds_s

    
class FinModel1(LightningModule):
    
    def __init__(self, cfg 
#                  conv_args = {}, mask_args = {}, segm_enc_args = {}, 
#                  segm_predictor_args = {}, 
#                  loss_args = {"n_negatives": 1, 
#                               "loss_args": {"reduction": "mean"}}, num_epoch = 2
                ):
        super(FinModel1, self).__init__()
        self.cfg = cfg
        self.conv_encoder = ConvFeatureEncoder(**cfg.conv_args)
#         self.projection = nn.Linear(256, 64) 
        # В работе https://github.com/felixkreuk/UnsupSeg projection улучшило r метрику за счет снижения размерности при работе с 
        # триплетами
        # Так как данная работа изначально основана на другой архитектуре, снижение размерности под комментарием
        
        self.frame_predictor = NegativeSampler(**cfg.loss_args)
        self.conv_layers_list = [[i[2], i[3]] for i in self.conv_encoder.conv_layers_list]
        self.segment_mean = SegmentsRepr(**cfg.mask_args)
        self.attention_calc = AttentionCalc()
#         self.segment_encoder = SegmentsEncoder(**cfg.segm_enc_args)
#         self.segment_recurrent = SegmentPredictor(**cfg.segm_predictor_args)
#         self.segment_predictor = NegativeSampler(**cfg.loss_args)
        self.num_epoch = cfg.num_epoch
        self.metr = RMetrics()
        self.unsup_loss = UnsupLoss()
        self.bounds = GetBound()
        
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.cfg.optimizer == "sgd":
            self.opt = optim.SGD(parameters, lr=self.cfg.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif self.cfg.optimizer == "adam":
            self.opt = optim.Adam(parameters, lr=self.cfg.learning_rate, weight_decay=5e-4)
        elif self.cfg.optimizer == "ranger":
            self.opt = optim_extra.Ranger(parameters, lr=self.cfg.learning_rate, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), 
                                          eps=1e-5,
                                          weight_decay=0)
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.opt}")
        self.scheduler = optim.lr_scheduler.StepLR(self.opt,
                                                   step_size=self.cfg.lr_anneal_step,
                                                   gamma=self.cfg.lr_anneal_gamma)
        
        return [self.opt]    
        
    def forward(self, x, attention_mask, return_secs):
        x = self.conv_encoder(x)
        x = x.transpose(1, 2)
        
        attention_mask1 = self.attention_calc.get_feature_vector_attention_mask(x.shape[1], attention_mask, 
                                                                                conv_layers=self.conv_layers_list)
        x, targets_frames, negatives_frames = self.frame_predictor(x, transpose=False, attention_mask=attention_mask1)
        frame_loss, frame_acc_score = self.frame_predictor.loss(x, targets_frames, negatives_frames, attention_mask=attention_mask1)
        
        frames_1_minus = x.roll(1, dims = 1)    
        preds = torch.cosine_similarity(x, frames_1_minus, dim=-1)[:, 1:]
        secs_preds = self.bounds.get_boundaries(preds)
        
        if return_secs:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              secs_pred = secs_pred,
                              hidden_states = x)
        else:
            return loss, dict(frame_loss=frame_loss.item(), 
                              frame_acc_score=frame_acc_score,
                              hidden_states = x
                             )
    
    def compute_all(self, x, secs, num_epoch, attention_mask, spectral_size,
                    return_secs = False):
        
        seq_len = x.shape[1]
        x = self.conv_encoder(x)
        x = x.transpose(1, 2)
#         x = self.projection(x)

        attention_mask1 = self.attention_calc.get_feature_vector_attention_mask(x.shape[1], attention_mask, 
                                                                                conv_layers=self.conv_layers_list)

        x, targets_frames, negatives_frames = self.frame_predictor(x, transpose=False, attention_mask=attention_mask1)
        frame_loss, frame_acc_score = self.frame_predictor.loss(x, targets_frames, negatives_frames, attention_mask=attention_mask1)
        frames_1_minus = x.roll(1, dims = 1)    
        preds = torch.cosine_similarity(x, frames_1_minus, dim=-1)[:, 1:]
    
        secs_preds = self.bounds.get_boundaries(preds)
        
        outsize, totstride, RFsize, ms_per_frame, ms_stride = self.metr.calculate_stride(seq_len, self.conv_layers_list)
        frames_true = self.metr.get_frames(secs, totstride)
        frames_pred = self.metr.get_frames(secs_preds, totstride)
        
        precision_counter, recall_counter, pred_counter, gt_counter = self.metr.get_stats(frames_true, frames_pred)
        precision, recall, f1, r_metr = self.metr.calc_metr(precision_counter, recall_counter, pred_counter, gt_counter)

        loss = frame_loss
        
        if return_secs:
            return loss, dict(frame_loss=frame_loss.item(), 
#                               frame_acc_score=frame_acc_score,
#                               segment_loss = segment_loss.item(), 
#                               segment_acc_score = segment_acc_score, 
                              loss = loss.item(), 
                              precision = precision, 
                              recall = recall,
                              f1 = f1,
                              r_metr = r_metr,
                              secs_pred = secs_pred)
        else:
            return loss, dict(frame_loss=frame_loss.item(), 
#                               frame_acc_score=frame_acc_score,
#                               segment_loss = segment_loss.item(), 
#                               segment_acc_score = segment_acc_score, 
                              loss = loss.item(), 
                              precision = precision, 
                              recall = recall,
                              f1 = f1,
                              r_metr = r_metr)

    def training_step(self, batch, batch_idx):

        loss, res_dict = self.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device),
                                                       spectral_size = batch['spectral_size'])
        self.log("train_loss", loss)
        for key, value in res_dict.items():
            self.log(key, value)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, res_dict = self.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device),
                                                       spectral_size = batch['spectral_size'])
        self.log("val_loss", loss)
        for key, value in res_dict.items():
            self.log('val_'+key, value)
        return loss
