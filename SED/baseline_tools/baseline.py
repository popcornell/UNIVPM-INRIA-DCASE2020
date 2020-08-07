from torch import nn

from .models.CRNN import CRNN
from .pcen import PCENstack


import torch
from SED.baseline_tools.loading import _load_scaler, _load_crnn
from SED.baseline_tools.utilities.utils import weights_init

class SEDBaseline(nn.Module):

    def __init__(self, confs, load_pretrained=False, use_init=False, scaling="Scaler", use_pcen=True, n_pcen=4, adversarial=False, freeze_bn=False):
        super(SEDBaseline, self).__init__()

        assert scaling in ["Scaler", "min-max", "cmvn", 'global-min-max']
        self.scaling = scaling if not use_pcen else 'global-min-max'  # FIXME cannot overwrite, but better default
        self.use_pcen = use_pcen
        self.adversarial = adversarial
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn


        n_layers = 7
        if not use_pcen:
            n_channel = 1
        else:
            n_channel = n_pcen
        crnn_kwargs = {"n_in_channel": n_channel, "nclass": confs["data"]["n_classes"] , "attention": True, "n_RNN_cell": 128,
                           "n_layers_RNN": 2,
                           "activation": "glu",
                           "dropout": 0.5,
                           "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                           "nb_filters": [16, 32, 64, 128, 128, 128, 128],
                           "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], "adversarial": adversarial}
       #CRNN(**crnn_kwargs)

        state = torch.load(confs["data"]["baseline_ckpt"])
        if load_pretrained:
            self.crnn = _load_crnn(state)
        else:
            self.crnn = CRNN(**crnn_kwargs) #
            if use_init:
                self.crnn.apply(weights_init)

        if self.scaling == "Scaler":
            scaler = _load_scaler(state)
            self.register_buffer("mean", torch.from_numpy(scaler.mean_).float())
            self.register_buffer("std", torch.from_numpy(scaler.std_).float())

        if self.use_pcen:
            self.pcen = PCENstack(n_pcen=n_pcen, in_f_size=128, s=[0.015, 0.08])


    def forward(self, x):
        #self.crnn.eval()
        #x = x.transpose(1, -1)

        #x = amp_to_db(x)

        # x is batch, 1, frames, channels

        if self.scaling == "Scaler":
            x = (x - self.mean) / (self.std + 1e-8)
        elif self.scaling == "min-max":
            x = (x - torch.min(x, dim=-1, keepdim=True)[0])  / (torch.max(x, dim=-1, keepdim=True)[0] - torch.min(x, dim=-1, keepdim=True)[0] + 1e-8)

        elif self.scaling == "cmvn":
            x = (x - torch.mean(x, dim=([x for x in range(len(x.shape)) if x != 0]), keepdim=True)) / (
                        torch.std(x, dim=([x for x in range(len(x.shape)) if x != 0]), keepdim=True) + 1e-8)
        elif self.scaling == "global-min-max":
            min_x = torch.min(torch.min(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            max_x = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            x = (x - min_x) / (max_x - min_x + 1e-8)

        elif self.scaling == "none":
            pass

        else:
            raise EnvironmentError

        if self.use_pcen:
            x = self.pcen(x.transpose(2, -1)).transpose(2, -1) # pcen wants batch, 1 , freq, time
        return self.crnn(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(SEDBaseline, self).train(mode)
        # if self.freeze_bn:
        #     print("Freezing Mean/Var of BatchNorm2D.")
        #     if self.freeze_bn_affine:
        #         print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

