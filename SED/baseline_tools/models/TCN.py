from torch import nn
import torch


EPS = 1e-8


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())

class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].
    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from
            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm
    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """
    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation,):
        super(Conv1DBlock, self).__init__()
        conv_norm = GlobLN
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class TCNAdv(nn.Module):
    # n blocks --> receptive field increases , n_repeats increases capacity mostly
    def __init__(self, in_chan=40, n_src=1, n_blocks=5, n_repeats=2,
                 bn_chan=64, hid_chan=128, kernel_size=3, attention=True
                 ):
        super(TCNAdv, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2 ** x))
        out_conv = nn.Conv1d(bn_chan, n_src, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        self.attention = attention
        if attention:
            self.dense_softmax = nn.Linear(bn_chan, n_src)
            self.softmax = nn.Softmax(-1)

        # Get activation function.

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
               [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """

        mixture_w = mixture_w.squeeze(1).transpose(1, -1)  # we do not need 3D input
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        strong = self.out(output).mean(-1)


        return strong.squeeze(-1)

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


class TCN(nn.Module):
    # n blocks --> receptive field increases , n_repeats increases capacity mostly
    def __init__(self, in_chan=40, n_src=1, n_blocks=5, n_repeats=2,
                 bn_chan=64, hid_chan=128, kernel_size=3, attention=True
                 ):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size

        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x))
        out_conv = nn.Conv1d(bn_chan, n_src, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        self.attention = attention
        if attention:
            self.dense_softmax = nn.Linear(bn_chan, n_src)
            self.softmax = nn.Softmax(-1)

        # Get activation function.

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
               [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        mixture_w = mixture_w.squeeze(1).transpose(1, -1) # we do not need 3D input
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
        strong = self.out(output).transpose(1, -1)
        logits = strong
        strong = nn.functional.sigmoid(strong)
        
        if self.attention:
            
            sof = self.dense_softmax(output.transpose(1, -1))
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong*sof).sum(1) / sof.sum(1)
        else:
            weak = strong.mean(1)


        return strong, weak, logits

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
        }
        return config


class SingleRNN(nn.Module):
    """ Module for a RNN block.
    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.
    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1,
                 dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                         num_layers=n_layers,
                                         batch_first=True,
                                         bidirectional=bool(bidirectional))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.

        rnn_output, _ = self.rnn(inp.transpose(1, -1))
        rnn_output = self.dropout(rnn_output)

        return rnn_output.transpose(1, -1)


class StackedResidualRNN(nn.Module):
    """ Stacked RNN with builtin residual connection.
    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, in_chan=40, n_src=1,  bn_chan=64, rnn_type="lstm", n_units=128, n_layers=3, dropout=0.2,
                 bidirectional=False, attention=True):
        super(StackedResidualRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is False, "Bidirectional not supported yet"
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList()

        self.bottleneck= nn.Sequential(GlobLN(in_chan), nn.Conv1d(in_chan, bn_chan, 1))

        for _ in range(n_layers):
            layer = nn.Sequential(SingleRNN(rnn_type, input_size=bn_chan,
                                         hidden_size=n_units,
                                         bidirectional=bidirectional, dropout=dropout),
                                  nn.Conv1d(n_units, bn_chan, 1), GlobLN(bn_chan))
            self.layers.append(layer)

        self.final = nn.Sequential(nn.PReLU(), nn.Conv1d(bn_chan, n_src, 1), nn.Sigmoid())
        self.attention = attention
        if attention:
            self.dense_softmax = nn.Sequenntial(nn.Linear(bn_chan, n_src))
            self.softmax = nn.Softmax(-1)

    def forward(self, x):
        """ Builtin residual connections + dropout applied before residual.
            Input shape : [batch, time_axis, feat_axis]
        """
        x = x.squeeze(1).transpose(1, -1)
        x = self.bottleneck(x)
        for rnn in self.layers:
            rnn_out = rnn(x)
            x = x + rnn_out

        strong = self.final(x).transpose(1, -1)
        if self.attention:
            sof = self.dense_softmax(x.transpose(1, -1))
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong* sof).sum(1) / sof.sum(1)
        else:
            weak = strong.mean(1)

        return strong, weak


if __name__ == "__main__":
    import torch
    inp = torch.rand((2, 1, 16000, 64))
    tcn = TCN(64, 5)
    out = tcn(inp)
    import ipdb
    ipdb.set_trace()
    print(out[0].shape)
