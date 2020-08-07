import torch
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
from .feats_utils import downsample, get_feats, choose_feats, normalize, choose_background
from .augmentations import apply_augmentation_chain

class DynamixMixUnlabel(Dataset):
    def __init__(self, jams, config, encoder, backgrounds, rirs, return_sources=False, as_labelled=False):

        """ mix sources and backgrounds dynamically"""

        self.jams = jams
        self.config = config
        self.encoder = encoder
        self.backgrounds = backgrounds
        self.rirs = rirs
        self.feats_func = choose_feats(config["feats"])
        self.as_labelled = as_labelled
        self.return_sources = return_sources

        # we construct a list of all sources

    def __len__(self):
        return len(self.jams)

    def read_foregrounds_jam(self, c_ex):
        foregrounds = []
        for s in c_ex:
            tmp, fs = sf.read(s)
            if len(tmp.shape) > 1:
                tmp = tmp[:, np.random.randint(0, tmp.shape[-1] - 1)]
            tmp = downsample(tmp, fs)
            tmp = tmp - np.mean(tmp)  # zero mean
            foregrounds.append(tmp)

        # we have a list of audios

        # apply time domain augmentation to each foreground separately
        foregrounds_weak = []
        foregrounds_strong = []
        min_lvl_weak = np.inf
        min_lvl_strong = np.inf
        first_weak = None
        first_strong = None

        orig_len = int(self.config["data"]["sample_rate"] * self.config["data"]["max_len_seconds"])

        for f in foregrounds:
            tmp = apply_augmentation_chain(f, self.config["augmentations"]["synth_unlabeled"]["weak_time"], self.rirs)
            if not first_weak:
                c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
            else:
                c_lvl = np.clip(np.random.normal(first_weak - 4, 3), max(first_weak - 20, -45),
                                    min(first_weak + 20, 0))
            tmp = normalize(tmp, c_lvl)
            min_lvl_weak = min(min_lvl_weak, c_lvl)
            tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
            foregrounds_weak.append(tmp)

            tmp = apply_augmentation_chain(f, self.config["augmentations"]["synth_unlabeled"]["strong_time"],
                                               self.rirs)
            if not first_strong:
                c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
            else:
                c_lvl = np.clip(np.random.normal(first_strong - 4, 3), max(first_strong - 20, -45),
                                    min(first_strong + 20, 0))

            tmp = normalize(tmp, c_lvl)
            min_lvl_strong = min(min_lvl_strong, c_lvl)
            tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
            foregrounds_strong.append(tmp)

        return foregrounds_weak, min_lvl_weak, foregrounds_strong, min_lvl_strong

    def get_labels(self, labels):

        return self.encoder.encode_strong_df(labels)

    def __getitem__(self, item):

        n_sources = np.random.randint(1, self.config["data"]["max_n_sources"]) # current number of sources
        # we sample randomly n_sources from self.jams
        indexes = np.arange(0, len(self.jams)) #[x for x in np.random.randint(0, len(self.jams))]
        indexes = np.random.choice(indexes, n_sources, replace=False)
        sources = [self.jams[indx] for indx in indexes]

        foregrounds_weak, min_weak, foregrounds_strong, min_strong = self.read_foregrounds_jam([x[0] for x in sources]) # IF YOU GET ERROR HERE DELETE CACHED

        background_weak = choose_background(self.backgrounds, None)  # do not augment backgrounds
        background_strong = choose_background(self.backgrounds, None)

        # normalize backgrounds
        background_weak = normalize(background_weak, np.clip(np.random.normal(-30, 12), -50, min_weak + 5))
        background_strong = normalize(background_strong, np.clip(np.random.normal(-30, 12), -50, min_strong + 5))
        mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)

        if self.return_sources:
            sources_weak = foregrounds_weak

        if np.max(np.abs(mixture_weak)) > 1:
            gain_weak = np.max(np.abs(mixture_weak))
            mixture_weak = mixture_weak / gain_weak
            if self.return_sources:
                sources_weak = [x / gain_weak for x in sources_weak]

        mixture_strong = np.sum(np.stack([*foregrounds_strong, background_strong]), 0)

        if self.return_sources:
            sources_strong = foregrounds_strong

        if np.max(np.abs(mixture_strong)) > 1:
            gain_strong = np.max(np.abs(mixture_strong))
            mixture_strong = mixture_strong / gain_strong
            if self.return_sources:
                sources_strong = [x / gain_strong for x in sources_strong]


        target_frames = self.config["feats"]["max_len"]
        mixture_weak = get_feats(self.feats_func, mixture_weak, target_frames)
        mixture_strong = get_feats(self.feats_func, mixture_strong, target_frames)

        if self.return_sources:

            labels = [x[-1] for x in sources]
            factor = (self.config["feats"]["hop_size"] / self.config["data"]["sample_rate"]) * self.config["net"][
                "pool_factor"]
            labels = [[z, int(x / factor), int(np.ceil(y / factor))] for z, x, y in labels]

            label_set = {}
            for entry in labels:
                if entry[0] not in label_set.keys():
                    label_set[entry[0]] = [entry]
                else:
                    label_set[entry[0]].append(entry)

            strong = []
            weak = []
            for k in label_set.keys():
                tmp = self.get_labels(label_set[k])
                strong.append(tmp)
                weak.append(np.sum(tmp, 0) >= 1)

            strong = np.array(strong)
            weak = np.array(weak)

            pad = np.zeros((self.config["data"]["max_n_sources"] - len(weak), strong.shape[1], strong.shape[-1]))
            strong = np.concatenate((strong, pad), 0)
            strong = torch.from_numpy(strong).float()

            pad = np.zeros((self.config["data"]["max_n_sources"] - len(weak), weak.shape[-1]))
            weak = np.concatenate((weak, pad), 0)
            weak = torch.from_numpy(weak).float()

            mask_weak = torch.zeros([1]).bool()
            if self.as_labelled:
                mask_strong = torch.ones([1]).bool()
            else:
                mask_strong = torch.zeros([1]).bool()

            if not sources_weak and not sources_strong:
                # empty
                sources_weak = torch.zeros((self.config["data"]["max_n_sources"],target_frames, mixture_weak.shape[0]))
                sources_strong = sources_weak
            else: # at least one source
                sources_weak = np.array([get_feats(self.feats_func, x, target_frames) for x in sources_weak])
                sources_strong = np.array([get_feats(self.feats_func, x, target_frames) for x in sources_strong])

                # we pad to max sources
                pad = np.zeros((self.config["data"]["max_n_sources"] - len(sources_weak) , sources_weak.shape[1], target_frames))
                sources_weak = np.concatenate((sources_weak, pad))
                sources_strong = np.concatenate((sources_strong, pad))

                sources_weak = torch.from_numpy(sources_weak).float().transpose(1, -1)
                sources_strong = torch.from_numpy(sources_strong).float().transpose(1, -1)

            mixture_weak = apply_augmentation_chain(mixture_weak,
                                                    self.config["augmentations"]["synth_unlabeled"]["weak_feats"])
            mixture_strong = apply_augmentation_chain(mixture_strong,
                                                      self.config["augmentations"]["synth_unlabeled"]["strong_feats"])

            mixture_weak = torch.from_numpy(mixture_weak.T).float()
            mixture_strong = torch.from_numpy(mixture_strong.T).float()

            return mixture_weak, sources_weak, mixture_strong, sources_strong, strong, weak, mask_strong, mask_weak

        else:

            labels = [x[-1] for x in sources]

            factor = (self.config["feats"]["hop_size"] / self.config["data"]["sample_rate"]) * self.config["net"][
                "pool_factor"]
            labels = [[z, int(x / factor), int(np.ceil(y / factor))] for z, x, y in labels]
            strong = self.get_labels(labels)
            weak = np.sum(strong, 0) >= 1
            weak = torch.from_numpy(weak).float()
            strong = torch.from_numpy(strong).float()

            mask_weak = torch.zeros([1]).bool()
            if self.as_labelled:
                mask_strong = torch.ones([1]).bool()
            else:
                mask_strong = torch.zeros([1]).bool()

            mixture_weak = apply_augmentation_chain(mixture_weak,
                                                self.config["augmentations"]["synth_unlabeled"]["weak_feats"])
            mixture_strong = apply_augmentation_chain(mixture_strong,
                                                  self.config["augmentations"]["synth_unlabeled"]["strong_feats"])

            mixture_weak = torch.from_numpy(mixture_weak.T).float()
            mixture_strong = torch.from_numpy(mixture_strong.T).float()

        return mixture_weak, mixture_strong, strong, weak, mask_strong, mask_weak


class DynamixMixLabel(Dataset):
    def __init__(self, jams, config, encoder, backgrounds, rirs, return_sources=False):

        """ mix sources and backgrounds dynamically"""

        self.jams = jams
        self.config = config
        self.encoder = encoder
        self.backgrounds = backgrounds
        self.rirs = rirs
        self.feats_func = choose_feats(config["feats"])
        self.return_sources = return_sources

        # we construct a list of all sources

    def __len__(self):
        return len(self.jams)

    def read_foregrounds_jam(self, c_ex):
        foregrounds = []
        for s in c_ex:
            tmp, fs = sf.read(s)
            if len(tmp.shape) > 1:
                tmp = tmp[:, np.random.randint(0, tmp.shape[-1] - 1)]
            tmp = downsample(tmp, fs)
            tmp = tmp - np.mean(tmp)  # zero mean
            foregrounds.append(tmp)

        # we have a list of audios

        # apply time domain augmentation to each foreground separately
        foregrounds_weak = []
        min_lvl_weak = np.inf
        first_weak = None

        orig_len = int(self.config["data"]["sample_rate"] * self.config["data"]["max_len_seconds"])

        for f in foregrounds:
            tmp = apply_augmentation_chain(f, self.config["augmentations"]["synth_unlabeled"]["weak_time"], self.rirs)
            if not first_weak:
                c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
            else:
                c_lvl = np.clip(np.random.normal(first_weak - 4, 3), max(first_weak - 20, -45),
                                    min(first_weak + 20, 0))
            tmp = normalize(tmp, c_lvl)
            min_lvl_weak = min(min_lvl_weak, c_lvl)
            tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
            foregrounds_weak.append(tmp)


        return foregrounds_weak, min_lvl_weak

    def get_labels(self, labels):

        return self.encoder.encode_strong_df(labels)

    def __getitem__(self, item):

        n_sources = np.random.randint(0, self.config["data"]["max_n_sources"]) # current number of sources
        # we sample randomly n_sources from self.jams
        indexes = np.arange(0, len(self.jams)) #[x for x in np.random.randint(0, len(self.jams))]
        indexes = np.random.choice(indexes, n_sources, replace=False)
        sources = [self.jams[indx] for indx in indexes]

        foregrounds_weak, min_weak = self.read_foregrounds_jam([x[0] for x in sources]) # read sources and return augmented ones

        background_weak = choose_background(self.backgrounds, None)  # do not augment backgrounds

        # normalize backgrounds
        background_weak = normalize(background_weak, np.clip(np.random.normal(-30, 12), -50, min_weak + 5))
        mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)

        if self.return_sources:
            sources_weak = foregrounds_weak

        if np.max(np.abs(mixture_weak)) > 1:
            gain_weak = np.max(np.abs(mixture_weak))
            mixture_weak = mixture_weak / gain_weak
            if self.return_sources:
                sources_weak = [x / gain_weak for x in sources_weak]




        target_frames = self.config["feats"]["max_len"]
        mixture_weak = get_feats(self.feats_func, mixture_weak, target_frames)


        if self.return_sources:

            raise NotImplementedError

            if not sources_weak:
                # empty
                sources_weak = torch.zeros((self.config["data"]["max_n_sources"], target_frames, mixture_weak.shape[0]))

            else: # at least one source
                sources_weak = np.array([get_feats(self.feats_func, x, target_frames) for x in sources_weak])


                # we pad to max sources
                pad = np.zeros((self.config["data"]["max_n_sources"] - len(sources_weak) , sources_weak.shape[1], target_frames))
                sources_weak = np.concatenate((sources_weak, pad))


                sources_weak = torch.from_numpy(sources_weak).float().transpose(1, -1)

            mixture_weak = apply_augmentation_chain(mixture_weak,
                                                    self.config["augmentations"]["synth_unlabeled"]["weak_feats"])

            mixture_weak = torch.from_numpy(mixture_weak.T).float()

            return mixture_weak, sources_weak, strong, weak, mask_strong, mask_weak

        else:

            labels = [x[-1] for x in sources]

            factor = (self.config["feats"]["hop_size"] / self.config["data"]["sample_rate"]) * self.config["net"][
                "pool_factor"]
            labels = [[z, int(x / factor), int(np.ceil(y / factor))] for z, x, y in labels]
            strong = self.get_labels(labels)
            weak = np.sum(strong, 0) >= 1
            weak = torch.from_numpy(weak).float()
            strong = torch.from_numpy(strong).float()

            mask_weak = torch.zeros([1]).bool()

            mask_strong = torch.ones([1]).bool()


            mixture_weak = apply_augmentation_chain(mixture_weak,
                                                self.config["augmentations"]["synth_unlabeled"]["weak_feats"])

            mixture_weak = torch.from_numpy(mixture_weak.T).float()

        return mixture_weak, strong, weak, mask_strong, mask_weak


class FussSet(Dataset):
    def __init__(self, fuss_sources, config, backgrounds, rirs, return_sources=True):

        """ mix sources and backgrounds dynamically"""

        self.fuss_sources = fuss_sources
        self.config = config
        self.backgrounds = backgrounds
        self.rirs = rirs
        self.feats_func = choose_feats(config["feats"])
        self.return_sources = return_sources

        # we construct a list of all sources

    def __len__(self):
        return len(self.fuss_sources)

    def read_foregrounds(self, c_ex):
        foregrounds = []
        for s in c_ex:
            tmp, fs = sf.read(s)
            if len(tmp.shape) > 1:
                tmp = tmp[:, np.random.randint(0, tmp.shape[-1] - 1)]
            tmp = downsample(tmp, fs)
            tmp = tmp - np.mean(tmp)  # zero mean
            foregrounds.append(tmp)

        # we have a list of audios

        # apply time domain augmentation to each foreground separately
        foregrounds_weak = []
        foregrounds_strong = []
        min_lvl_weak = np.inf
        min_lvl_strong = np.inf
        first_weak = None
        first_strong = None

        orig_len = int(self.config["data"]["sample_rate"] * self.config["data"]["max_len_seconds"])

        for f in foregrounds:
            tmp = apply_augmentation_chain(f, self.config["augmentations"]["synth_unlabeled"]["weak_time"], self.rirs)
            if not first_weak:
                c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
            else:
                c_lvl = np.clip(np.random.normal(first_weak - 4, 3), max(first_weak - 20, -45),
                                    min(first_weak + 20, 0))
            tmp = normalize(tmp, c_lvl)
            min_lvl_weak = min(min_lvl_weak, c_lvl)
            tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
            foregrounds_weak.append(tmp)

            tmp = apply_augmentation_chain(f, self.config["augmentations"]["synth_unlabeled"]["strong_time"],
                                               self.rirs)
            if not first_strong:
                c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
            else:
                c_lvl = np.clip(np.random.normal(first_strong - 4, 3), max(first_strong - 20, -45),
                                    min(first_strong + 20, 0))

            tmp = normalize(tmp, c_lvl)
            min_lvl_strong = min(min_lvl_strong, c_lvl)
            tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
            foregrounds_strong.append(tmp)

        return foregrounds_weak, min_lvl_weak, foregrounds_strong, min_lvl_strong


    def __getitem__(self, item):

        n_sources = np.random.randint(1, self.config["data"]["max_n_sources"]) # current number of sources
        # we sample randomly n_sources from self.jams
        indexes = np.arange(0, len(self.fuss_sources)) #[x for x in np.random.randint(0, len(self.jams))]
        indexes = np.random.choice(indexes, n_sources, replace=False)
        sources = [self.fuss_sources[indx] for indx in indexes]

        foregrounds_weak, min_weak, foregrounds_strong, min_strong = self.read_foregrounds(sources) # IF YOU GET ERROR HERE DELETE CACHED

        background_weak = choose_background(self.backgrounds, None)  # do not augment backgrounds
        background_strong = choose_background(self.backgrounds, None)

        # normalize backgrounds
        background_weak = normalize(background_weak, np.clip(np.random.normal(-30, 12), -50, min_weak + 5))
        background_strong = normalize(background_strong, np.clip(np.random.normal(-30, 12), -50, min_strong + 5))
        mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)

        if self.return_sources:
            sources_weak = foregrounds_weak

        if np.max(np.abs(mixture_weak)) > 1:
            gain_weak = np.max(np.abs(mixture_weak))
            mixture_weak = mixture_weak / gain_weak
            if self.return_sources:
                sources_weak = [x / gain_weak for x in sources_weak]

        mixture_strong = np.sum(np.stack([*foregrounds_strong, background_strong]), 0)

        if self.return_sources:
            sources_strong = foregrounds_strong

        if np.max(np.abs(mixture_strong)) > 1:
            gain_strong = np.max(np.abs(mixture_strong))
            mixture_strong = mixture_strong / gain_strong
            if self.return_sources:
                sources_strong = [x / gain_strong for x in sources_strong]


        target_frames = self.config["feats"]["max_len"]
        mixture_weak = get_feats(self.feats_func, mixture_weak, target_frames)
        mixture_strong = get_feats(self.feats_func, mixture_strong, target_frames)

        if not sources_weak and not sources_strong:
            # empty
            sources_weak = torch.zeros((self.config["data"]["max_n_sources"],target_frames, mixture_weak.shape[0]))
            sources_strong = sources_weak
        else: # at least one source
            sources_weak = np.array([get_feats(self.feats_func, x, target_frames) for x in sources_weak])
            sources_strong = np.array([get_feats(self.feats_func, x, target_frames) for x in sources_strong])

            # we pad to max sources
            pad = np.zeros((self.config["data"]["max_n_sources"] - len(sources_weak) , sources_weak.shape[1], target_frames))
            sources_weak = np.concatenate((sources_weak, pad))
            sources_strong = np.concatenate((sources_strong, pad))

            sources_weak = torch.from_numpy(sources_weak).float().transpose(1, -1)
            sources_strong = torch.from_numpy(sources_strong).float().transpose(1, -1)

            mixture_weak = apply_augmentation_chain(mixture_weak,
                                                    self.config["augmentations"]["synth_unlabeled"]["weak_feats"])
            mixture_strong = apply_augmentation_chain(mixture_strong,
                                                      self.config["augmentations"]["synth_unlabeled"]["strong_feats"])

            mixture_weak = torch.from_numpy(mixture_weak.T).float()
            mixture_strong = torch.from_numpy(mixture_strong.T).float()

            max_len_targets = self.config["feats"]["max_len"] // self.config["net"]["pool_factor"]
            strong = torch.zeros(max_len_targets, self.config["data"]["n_classes"])
            weak = torch.zeros(self.config["data"]["n_classes"])

            mask_weak = torch.zeros([1]).bool()
            mask_strong = torch.zeros([1]).bool()

            return mixture_weak, sources_weak, mixture_strong, sources_strong, strong, weak, mask_strong, mask_weak