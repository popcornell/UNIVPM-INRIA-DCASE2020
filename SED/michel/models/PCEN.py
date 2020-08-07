import torch
import torch.nn as nn
import numpy as np
import scipy.signal


class PCENstack(nn.Module):
    """Computes a learnable stack of PCEN transformations of an input spectrogram.

        Parameters alpha, delta and r are directly optimized, as well as the vector weights for
        the desired smoothers.

    """

    def __init__(self, n_pcen, in_f_size, s, eps=1e-6, batch_norm=False):
        """Constructor. Initializes the PCEN transforms parameters to random values close to the default.

            n_pcen (int): Number of PCEN transform to use. (positive >= 1)
            in_f_size (int): Number of frequency channels in input representation
            s (list): Smoothers
            eps (float): Stability param for PCEN computation
            batch_norm (boolean): Normalization scheme
        """

        super(PCENstack, self).__init__()

        self.custom_initialization = True

        self.n_pcen = n_pcen


        if self.custom_initialization:

            if n_pcen == 4:

                # # Best params learned in previous training 3785: original depthwise cnn block
                # self.i_sig_alpha = torch.log(torch.tensor([0.84599406 / (1.0 - 0.84599406),
                #                                            0.7390965 / (1.0 - 0.7390965),
                #                                            0.6660008 / (1.0 - 0.6660008),
                #                                            0.58494717 / (1.0 - 0.58494717)]))
                # self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                # self.log_delta = torch.tensor([7.182488, 7.489906, 6.839884, 1.678105]).log_()
                # self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                # self.i_sig_r = torch.log(torch.tensor([0.6169229 / (1.0 - 0.6169229),
                #                                        0.5803292 / (1.0 - 0.5803292),
                #                                        0.53803605 / (1.0 - 0.53803605),
                #                                        0.72681004 / (1.0 - 0.72681004)]))
                # self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

                # # Best params learned in previous training 0742
                # self.i_sig_alpha = torch.log(torch.tensor([0.866621 / (1.0 - 0.866621),
                #                                            0.708792 / (1.0 - 0.708792),
                #                                            0.49426293 / (1.0 - 0.49426293),
                #                                            0.5945607 / (1.0 - 0.5945607)]))
                # self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                # self.log_delta = torch.tensor([8.618472, 9.501848, 8.4144125, 0.52592754]).log_()
                # self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                # self.i_sig_r = torch.log(torch.tensor([0.591497 / (1.0 - 0.591497),
                #                                        0.5063154 / (1.0 - 0.5063154),
                #                                        0.46923196 / (1.0 - 0.46923196),
                #                                        0.6866194 / (1.0 - 0.6866194)]))
                # self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

                # Best params learned in previous training 0742 with logmels
                self.i_sig_alpha = torch.log(torch.tensor([0.8925986 / (1.0 - 0.8925986),
                                                           0.7687151 / (1.0 - 0.7687151),
                                                           0.5038489 / (1.0 - 0.5038489),
                                                           0.3695674 / (1.0 - 0.3695674)]))
                self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                self.log_delta = torch.tensor([7.6649485, 9.902711, 10.697797, 1.9174507]).log_()
                self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                self.i_sig_r = torch.log(torch.tensor([0.5941242 / (1.0 - 0.5941242),
                                                       0.5078372 / (1.0 - 0.5078372),
                                                       0.44175446 / (1.0 - 0.44175446),
                                                       0.874569 / (1.0 - 0.874569)]))
                self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

            if n_pcen == 1:

                # Best params learned in previous training 2024
                self.i_sig_alpha = torch.log(torch.tensor([0.69014716 / (1.0 - 0.69014716)]))
                self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                self.log_delta = torch.tensor([2.7870882]).log_()
                self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                self.i_sig_r = torch.log(torch.tensor([0.6006879 / (1.0 - 0.6006879)]))
                self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)


            if n_pcen == 2:

                # # Best params learned in previous training 2024
                # self.i_sig_alpha = torch.log(torch.tensor([0.7717817 / (1.0 - 0.7717817),
                #                                            0.6286456 / (1.0 - 0.6286456)]))
                # self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                # self.log_delta = torch.tensor([9.098704, 3.8612893]).log_()
                # self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                # self.i_sig_r = torch.log(torch.tensor([0.48453242 / (1.0 - 0.48453242),
                #                                        0.61920357 / (1.0 - 0.61920357)]))
                # self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

                # # Best params learned in previous training 0742
                # self.i_sig_alpha = torch.log(torch.tensor([0.79684514 / (1.0 - 0.79684514),
                #                                            0.6123944 / (1.0 - 0.6123944)]))
                # self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                # self.log_delta = torch.tensor([9.68656, 1.8009254]).log_()
                # self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                # self.i_sig_r = torch.log(torch.tensor([0.46314678 / (1.0 - 0.46314678),
                #                                        0.6570781 / (1.0 - 0.6570781)]))
                # self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

                # Best params learned in previous training 0677 with logmels 
                self.i_sig_alpha = torch.log(torch.tensor([0.8186727 / (1.0 - 0.8186727),
                                                           0.39384758 / (1.0 - 0.39384758)]))
                self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                self.log_delta = torch.tensor([10.689431, 3.4707892]).log_()
                self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                self.i_sig_r = torch.log(torch.tensor([0.44950205 / (1.0 - 0.44950205),
                                                       0.75343627 / (1.0 - 0.75343627)]))
                self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)


            if n_pcen == 5:

                # Best params learned in previous training 2024
                self.i_sig_alpha = torch.log(torch.tensor([0.61010957 / (1.0 - 0.61010957),
                                                           0.82052416 / (1.0 - 0.82052416),
                                                           0.62170863 / (1.0 - 0.62170863),
                                                           0.7137496 / (1.0 - 0.7137496),
                                                           0.71177524 / (1.0 - 0.71177524)]))
                self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                self.log_delta = torch.tensor([7.5384, 8.178805, 7.3263173, 8.520552, 8.926732]).log_()
                self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                self.i_sig_r = torch.log(torch.tensor([0.54768777 / (1.0 - 0.54768777),
                                                       0.51798695 / (1.0 - 0.51798695),
                                                       0.53117794 / (1.0 - 0.53117794),
                                                       0.51186365 / (1.0 - 0.51186365),
                                                       0.49415016 / (1.0 - 0.49415016)]))
                self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)

            if n_pcen == 10:
                print("10 Layer PCEN Stack")
                # Best params learned in previous training 2024
                self.i_sig_alpha = torch.log(torch.tensor([0.70409125 / (1.0 - 0.70409125),
                                                           0.6647366 / (1.0 - 0.6647366),
                                                           0.7705285 / (1.0 - 0.7705285),
                                                           0.8275785 / (1.0 - 0.8275785),
                                                           0.6283538 / (1.0 - 0.6283538),
                                                           0.6891031 / (1.0 - 0.6891031),
                                                           0.79970217 / (1.0 - 0.79970217),
                                                           0.6423833 / (1.0 - 0.6423833),
                                                           0.8649959 / (1.0 - 0.8649959),
                                                           0.68174213 / (1.0 - 0.68174213)]))
                self.i_sig_alpha = nn.Parameter(self.i_sig_alpha, requires_grad=True)

                self.log_delta = torch.tensor([7.4051914, 9.165579, 13.206677, 9.980098, 8.893635,
                                               8.362788, 11.084309, 4.7399635, 11.553849, 7.1772466]).log_()
                self.log_delta = nn.Parameter(self.log_delta, requires_grad=True)

                self.i_sig_r = torch.log(torch.tensor([0.5475462 / (1.0 - 0.5475462),
                                                       0.48858947 / (1.0 - 0.48858947),
                                                       0.42170933 / (1.0 - 0.42170933),
                                                       0.49574375 / (1.0 - 0.49574375),
                                                       0.4880911 / (1.0 - 0.4880911),
                                                       0.5129925 / (1.0 - 0.5129925),
                                                       0.45981303 / (1.0 - 0.45981303),
                                                       0.6044327 / (1.0 - 0.6044327),
                                                       0.47022137 / (1.0 - 0.47022137),
                                                       0.54202306 / (1.0 - 0.54202306)]))
                self.i_sig_r = nn.Parameter(self.i_sig_r, requires_grad=True)


        else:

            # inverse_sigmoid(alpha), using the default value of alpha: 0.98
            self.i_sig_alpha = torch.log(torch.tensor(0.8 / (1.0 - 0.8)))
            # self.i_sig_alpha = nn.Parameter(self.i_sig_alpha * (1.0 + torch.rand(n_pcen) * 0.1))
            self.i_sig_alpha = nn.Parameter(self.i_sig_alpha * (1.0 + torch.rand(n_pcen) * 0.0001), requires_grad=True)
            # log(delta), using the default value of delta 2.0
            self.log_delta = torch.tensor(10.0).log_()
            # self.log_delta = nn.Parameter(self.log_delta * (1.0 + torch.rand(n_pcen) * 0.1))
            self.log_delta = nn.Parameter(self.log_delta * (1.0 + torch.rand(n_pcen) * 0.0001), requires_grad=True)
            # inverse_sigmoid(r), using the default value of r: 0.5
            self.i_sig_r = torch.tensor(0.0)
            # self.i_sig_r = nn.Parameter(self.i_sig_r * (1.0 + torch.rand(n_pcen) * 0.1))
            self.i_sig_r = nn.Parameter(self.i_sig_r * (1.0 + torch.rand(n_pcen) * 0.0001), requires_grad=True)


        self.s = s
        # weight vectors for smoothers
        self.z_ks = torch.randn((len(s), in_f_size)) * 0.1 + np.log(1 / len(s))
        self.z_ks = nn.Parameter(self.z_ks * (1.0 + torch.rand(n_pcen, self.z_ks.shape[0], self.z_ks.shape[-1]) * 0.0001), requires_grad=True)

        self.eps = eps
        
        
        # batch normalization
        if batch_norm:
            # self.normalization = nn.BatchNorm2d(n_pcen, eps=0.001, momentum=0.99)
            self.normalization = batch_norm
        else:
            self.normalization = False


    def forward(self, x):
        """Forward pass of the learnable PCEN stack
            :param x: Input. Shape [Batch, number of PCEN layer, Frequency, Time]
            :type x: torch.Tensor
            :return: Output.
            :rtype: torch.Tensor
        """

        # Expand params

        alpha = self.i_sig_alpha.sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
        delta = self.log_delta.exp().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
        r = self.i_sig_r.sigmoid().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
        z_ks = self.z_ks.permute(1, 0, -1)

        # map the weights stored in ]-inf, +inf[ to [0, 1] and such that they sum up to 1 (softmax)
        # print('z_ks size:', self.z_ks.size())
        
        # weight vectors
        w_ks = (z_ks.exp() / z_ks.exp().sum(dim=0)).unsqueeze(1).unsqueeze(-1).expand(-1, x.shape[0], -1, -1, x.shape[-1])
        # spectrogram smoothers
        smoothers = torch.stack([torch.tensor(scipy.signal.filtfilt([s], [1, s - 1], x.cpu(), axis=-1, padtype=None).astype(np.float32), device=x.device) for s in self.s])
        # smoother combination
        M = (smoothers * w_ks).sum(dim=0)


        # PCEN Equation
        M = torch.exp(-alpha * (float(np.log(self.eps)) + torch.log1p(M / self.eps)))
        pcen_stack = (x * M + delta).pow(r) - delta.pow(r)

        # Min-max normalization scheme
        if self.normalization:

            pcen_min = torch.min(pcen_stack.view(pcen_stack.size(0), pcen_stack.size(1), -1), -1).values
            pcen_min = pcen_min.unsqueeze(-1).unsqueeze(-1).expand(pcen_stack.shape[0], -1, pcen_stack.shape[2], pcen_stack.shape[3])

            pcen_max = torch.max(pcen_stack.view(pcen_stack.size(0), pcen_stack.size(1), -1), -1).values
            pcen_max = pcen_max.unsqueeze(-1).unsqueeze(-1).expand(pcen_stack.shape[0], -1, pcen_stack.shape[2], pcen_stack.shape[3])

            pcen_stack = (pcen_stack - pcen_min) / (pcen_max - pcen_min + 1e-6)


        return pcen_stack