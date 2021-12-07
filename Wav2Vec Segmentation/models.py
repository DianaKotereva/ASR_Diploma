
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
            n_in, n_out, k, stride, conv_bias = False
        ):
            def make_conv(n_in, n_out, k, stride=stride, bias=False):
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            return nn.Sequential(
                    make_conv(n_in, n_out, k, stride=stride, bias=conv_bias),
                    nn.BatchNorm1d(n_out),
                    nn.LeakyReLU()
                )
        
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (in_d, out_d, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    out_d,
                    k,
                    stride, 
                    conv_bias = conv_bias
                )
            )

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    
class SegmentRepresentation:
    
    def __init__
        
    def forward:
        

def boundary(frames, thres = 0.05):

#     [batch_size, channels, seq_len]
    
    indexes = torch.zeros(frames.shape[0], frames.shape[-1])
    indexes[:, 0] = 1

    frames_1_plus = frames.roll(-1, dims = 2)

    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos = cosine_sim(frames, frames_1_plus)[:, :-1]

    min_cos = torch.min(cos, dim=1).values
    min_cos = torch.cat([min_cos.unsqueeze(1)]*(frames.shape[-1]-1), dim = 1)
    max_cos = torch.max(cos, dim=1).values
    max_cos = torch.cat([max_cos.unsqueeze(1)]*(frames.shape[-1]-1), dim = 1)

    d = torch.ones(frames.shape[0], (frames.shape[-1])-1)-(cos - min_cos)/(max_cos-min_cos)
        
    d_1_plus = d.roll(-1, dims = 1)
    d_2_plus = d.roll(-2, dims = 1)
    d_1_minus = d.roll(1, dims = 1)
    d_2_minus = d.roll(2, dims = 1)
        
    zeros_d = torch.zeros(d.shape)

    p_1 = torch.min(torch.max((d-d_1_plus), zeros_d), torch.max((d-d_1_minus), zeros_d))
    p_1[:, 0] = 0
    p_1[:, -1] = 0
    p_2 = torch.min(torch.max((d-d_2_plus), zeros_d), torch.max((d-d_2_minus), zeros_d))
    p_2[:, [0, 1]] = 0
    p_2[:, [-2, -1]] = 0

    pt = torch.min(torch.max(torch.max(p_1, p_2)-torch.full(p_1.shape, thres), zeros_d), p_1)

    b_soft = torch.tanh(10*p_2)
    b_hard = torch.tanh(10000000*p_2)
    b = b_soft + (b_hard-b_soft).detach()
    
    return b
















