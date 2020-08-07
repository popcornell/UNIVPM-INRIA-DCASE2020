import numpy as np
from torch.utils.data import Dataset
from .feats_utils import choose_feats
from .feats_utils import downsample, get_feats
import soundfile as sf
import torch


class ValSet(Dataset):

    def __init__(self, strong_labels, confs, encoder, return_filename=True):
        super(ValSet, self).__init__()
        self.examples = strong_labels
        self.feats_func = choose_feats(confs["feats"])
        self.confs = confs
        self.encoder = encoder
        self.return_filename=return_filename

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, fs = sf.read(c_ex["mixture"])
        if len(mixture.shape) > 1:  # multi channel
            mixture = np.mean(mixture, axis=-1)
        # downsample if required
        mixture = downsample(mixture, fs)

        pad_to = self.confs["feats"]["max_len"]
        mixture = get_feats(self.feats_func, mixture, target_frames=pad_to)
        mixture = torch.from_numpy(mixture.T).float()

        # labels
        labels = c_ex["labels"]
        # check if labels exists:
        if not isinstance(labels[0][0], str):
            max_len_targets = self.confs["feats"]["max_len"] // self.confs["net"]["pool_factor"]
            strong = torch.zeros(max_len_targets, self.confs["data"]["n_classes"])
            weak = torch.zeros(self.confs["data"]["n_classes"])

        else:
            # to steps
            factor = (self.confs["feats"]["hop_size"] / self.confs["data"]["sample_rate"]) * self.confs["net"][
                "pool_factor"]
            labels = [[z, int(x / factor), int(np.ceil(y / factor))] for z, x, y in labels]
            strong = self.encoder.encode_strong_df(labels)
            weak = np.sum(strong, 0) >= 1
            weak = torch.from_numpy(weak).float()
            strong = torch.from_numpy(strong).float()

        if self.return_filename:
            return mixture, strong, weak, c_ex["mixture"]
        else:
            return mixture, strong, weak


class EvalSet(Dataset):

    def __init__(self, files, confs, encoder, return_filename=True):
        super(EvalSet, self).__init__()
        self.examples = files
        self.feats_func = choose_feats(confs["feats"])
        self.confs = confs
        self.encoder = encoder
        self.return_filename=return_filename

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, fs = sf.read(c_ex)
        if len(mixture.shape) > 1:  # multi channel
            mixture = np.mean(mixture, axis=-1)
        # downsample if required
        mixture = downsample(mixture, fs)

        pad_to = self.confs["feats"]["max_len"]
        mixture = get_feats(self.feats_func, mixture, target_frames=pad_to)
        mixture = torch.from_numpy(mixture.T).float()

        # labels
        #labels = c_ex["labels"]
        # check if labels exists:
        #if not isinstance(labels[0][0], str):
        max_len_targets = self.confs["feats"]["max_len"] // self.confs["net"]["pool_factor"]
        strong = torch.zeros(max_len_targets, self.confs["data"]["n_classes"])
        weak = torch.zeros(self.confs["data"]["n_classes"])


        if self.return_filename:
            return mixture, strong, weak, c_ex
        else:
            return mixture, strong, weak
