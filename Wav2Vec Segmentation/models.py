import torch
from torch import nn
from utils import ConstrativeLoss, sample_negatives

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
    
    
class SegmentsRepr(nn.Module):
    def __init__(self, thres = 0.05):
        super(SegmentsRepr, self).__init__()
        self.thres = thres

    def boundary(self, frames):

        #batch_size x seq_len x dim
        indexes = torch.zeros(frames.shape[0], frames.shape[1])
        indexes[:, 0] = 1

        frames_1_plus = frames.roll(-1, dims = 1)

        cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        cos = cosine_sim(frames, frames_1_plus)[:, :-1]
        
        min_cos = torch.min(cos, dim=1).values
        min_cos = min_cos.unsqueeze(1).expand(min_cos.shape[0], frames.shape[1]-1)
        max_cos = torch.max(cos, dim=1).values
        max_cos = max_cos.unsqueeze(1).expand(max_cos.shape[0], frames.shape[1]-1)
        
        d = torch.ones(frames.shape[0], (frames.shape[1])-1)-(cos - min_cos)/(max_cos-min_cos)

        d_1_plus = d.roll(-1, dims = 1)
        d_2_plus = d.roll(-2, dims = 1)
        d_1_minus = d.roll(1, dims = 1)
        d_2_minus = d.roll(2, dims = 1)

        zeros_d = torch.zeros(d.shape)

        p_1 = torch.min(torch.max((d-d_1_plus), zeros_d), torch.max((d-d_1_minus), zeros_d))
        p_1[:, [0, 1]] = 0
#         p_1[:, -1] = 0
        p_2 = torch.min(torch.max((d-d_2_plus), zeros_d), torch.max((d-d_2_minus), zeros_d))
        p_2[:, [0, 1]] = 0
        p_2[:, [-2, -1]] = 0

        pt = torch.min(torch.max(torch.max(p_1, p_2)-torch.full(p_1.shape, self.thres), zeros_d), p_1)

        b_soft = torch.tanh(10*p_2)
        b_hard = torch.tanh(10000000*p_2)
        b = b_soft + (b_hard-b_soft).detach()
        
        indexes[:, 1:] = b

        return indexes
    
    def receive_mask(self, b):
#         print(b.shape)
        b_cumsum = torch.cumsum(b, axis=1)
        num_fr = torch.sum(b, axis=1)
        
        # Создать arange
        c = torch.arange(1, int(torch.max(num_fr))+1).unsqueeze(0).unsqueeze(0).expand(b_cumsum.shape[0], b_cumsum.shape[1], int(torch.max(num_fr)))
#         print(c.shape)
        maxs = num_fr.unsqueeze(-1).unsqueeze(-1).expand(c.shape[0], c.shape[1], c.shape[-1])
        cc = c.clone()
        cc[cc>maxs] = 0 # Проверить, пойдут ли градиенты через эти ворота
        
        # Получение индексов
        b_cumsum1 = b_cumsum.unsqueeze(-1).expand(b_cumsum.shape[0], b_cumsum.shape[1], cc.shape[-1])
        unres = torch.tanh(10*torch.abs(cc - b_cumsum1))
        res = torch.ones(unres.shape)-unres
        
        # Нормирование
        sums = torch.sum(res, dim=1).unsqueeze(1).expand(res.shape)+unres # + unres чтобы не было null значений на нулях
        masks = res/sums
        masks = masks.transpose(2, 1)
        
        return masks
    
    def forward(self, frames):
        # Получение сегментов
        b = self.boundary(frames)
        mask = self.receive_mask(b)
        
#         print('Mask', mask.shape)
#         print('Frames', frames.shape)
        segments = torch.bmm(mask, frames)
        return segments


class SegmentsEncoder(nn.Module):
    
    def __init__(self, num_feats = 256):
        super(SegmentsEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(num_feats, num_feats),
                                 nn.LeakyReLU(),
                                 nn.Linear(num_feats, num_feats)
                                   )
        
    def forward(self, x):
        # batch_size, seq_len, dim
        return self.enc(x)
    
    
class EncoderModel(nn.Module):
    
    def __init__(self, encoder, n_negatives = 10, loss_args = {'reduction': 'sum'}):
        super(EncoderModel, self).__init__()
        self.conv_encoder = encoder
        self.n_negatives = n_negatives
        self.loss = ConstrativeLoss(**loss_args)
        
    def forward(self, x, transpose = False):
        x = self.conv_encoder(x)
        if transpose:
#             print('Transpose')
            x = x.transpose(1, 2)
        targets = torch.roll(x, -1, dims=1)
        
        negatives, negs_ids = sample_negatives(targets, n_negatives = self.n_negatives)
        return x, targets, negatives
        
    def compute_all(self, batch, transpose = False):
        
        x, targets, negatives = self.forward(batch, transpose)
        loss, acc_score = self.loss(x, targets, negatives)
        return loss, dict(loss=loss.item(), 
                          acc=acc_score)


class SegmentPredictor(nn.Module):
    
    def __init__(self, input_size = 256, hidden_size = 64, output_size = 256):
        super(SegmentPredictor, self).__init__()
        self.rnn = nn.GRU(input_size = input_size, 
                          hidden_size = hidden_size, 
                          batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
#         print('GRU', x.shape)
        x, _ = self.rnn(x)
#         print('GRU end', x.shape)
        x = self.linear(x)
#         print('Linear', x.shape)
        return x


class FinModel(nn.Module):
    
    def __init__(self, conv_args = {}, mask_args = {}, segm_enc_args = {}, 
                 segm_predictor_args = {}, 
                 loss_args = {"n_negatives": 10, 
                              "loss_args": {"reduction": "mean"}}, num_epoch = 2):
        super(FinModel, self).__init__()
        self.frame_predictor = EncoderModel(ConvFeatureEncoder(**conv_args), **loss_args)
        self.segment_mean = SegmentsRepr(**mask_args)
        self.segment_encoder = SegmentsEncoder(**segm_enc_args)
        self.segment_predictor = EncoderModel(SegmentPredictor(**segm_predictor_args), **loss_args)
        self.num_epoch = num_epoch
        
    def forward(self, x):
#         print('Start', x.shape)
        x, targets_frames, negatives_frames = self.frame_predictor(x, transpose = True)
#         print(x.shape)
        x = self.segment_mean(x)
#         print(x.shape)
        x = self.segment_encoder(x)
#         print(x.shape)
        x, targets_segments, negatives_segments = self.segment_predictor(x)
        return x
    
    def compute_all(self, x, num_epoch):
        
        x, targets_frames, negatives_frames = self.frame_predictor(x, transpose = True)
        frame_loss, frame_acc_score = self.frame_predictor.loss(x, targets_frames, negatives_frames)
        
        x = self.segment_mean(x)
        x = self.segment_encoder(x)
        
        x, targets_segments, negatives_segments = self.segment_predictor(x)
        segment_loss, segment_acc_score = self.segment_predictor.loss(x, targets_segments, negatives_segments)
        
        if num_epoch >= self.num_epoch:
            loss = frame_loss+segment_loss
        else:
            loss = frame_loss
        
        return loss, dict(frame_loss=frame_loss.item(), 
                          frame_acc_score=frame_acc_score,
                          segment_loss = segment_loss.item(), 
                          segment_acc_score = segment_acc_score, 
                          loss = loss.item())



