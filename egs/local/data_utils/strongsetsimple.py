from torch.utils.data import Dataset
import soundfile as sf
import torch
import numpy as np
from .augmentations import apply_augmentation_chain
from .feats_utils import choose_feats
from .feats_utils import choose_background, downsample, normalize, get_feats
import copy


class StrongSetSimpleUnlabeled(Dataset):
    """we use the foregrounds and backgrounds created with Scaper but dynamically mix them and augment them"""

    def __init__(self, jams, cfg, encoder, time_augment=True, as_labelled=False,
                  backgrounds=None, rirs=None):

        self.jams = jams
        self.cfg = cfg
        self.time_augment = time_augment
        self.encoder = encoder
        # we get the single sources and augment them singularly
        # we get all backgrounds
        self.feats_func = choose_feats(cfg["feats"])
        self.rirs = rirs
        self.backgrounds = backgrounds
        self.as_labelled = as_labelled

    def __len__(self):
        return len(self.jams)

    def read_foregrounds_jam(self, c_ex):

        foregrounds = []
        for s in c_ex["foreground"]:
                tmp, fs = sf.read(s)
                if len(tmp.shape) > 1:
                    if self.time_augment:
                        tmp = tmp[:, np.random.randint(0, tmp.shape[-1]-1)]
                    else:
                        tmp = np.mean(tmp, -1)
                tmp = downsample(tmp, fs)
                tmp = tmp - np.mean(tmp) # zero mean
                foregrounds.append(tmp)

        # we have a list of audios
        if self.time_augment:
            # apply time domain augmentation to each foreground separately
            foregrounds_weak = []
            foregrounds_strong = []
            min_lvl_weak = np.inf
            min_lvl_strong = np.inf
            first_weak = None
            first_strong = None

            orig_len = int(self.cfg["data"]["sample_rate"]*self.cfg["data"]["max_len_seconds"])

            for f in foregrounds:
                tmp = apply_augmentation_chain(f, self.cfg["augmentations"]["synth_unlabeled"]["weak_time"], self.rirs)
                if not first_weak:
                    c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
                else:
                    c_lvl = np.clip(np.random.normal(first_weak - 4, 3), max(first_weak - 20, -45), min(first_weak + 20, 0))
                tmp = normalize(tmp, c_lvl)
                min_lvl_weak = min(min_lvl_weak, c_lvl)
                tmp = np.pad(tmp, (0, orig_len-len(tmp)), mode="constant")
                foregrounds_weak.append(tmp)

                tmp = apply_augmentation_chain(f, self.cfg["augmentations"]["synth_unlabeled"]["strong_time"], self.rirs)
                if not first_strong:
                    c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
                else:
                    c_lvl = np.clip(np.random.normal(first_strong - 4, 3), max(first_strong - 20, -45), min(first_strong + 20, 0))

                tmp = normalize(tmp, c_lvl)
                min_lvl_strong = min(min_lvl_strong, c_lvl)
                tmp = np.pad(tmp, (0, orig_len - len(tmp)), mode="constant")
                foregrounds_strong.append(tmp)

            return foregrounds_weak, min_lvl_weak, foregrounds_strong, min_lvl_strong
        else:
            return foregrounds

    def get_labels(self, labels):

        return self.encoder.encode_strong_df(labels)

    def __getitem__(self, item):

        c_ex = self.jams[item]

        foregrounds_weak, min_weak, foregrounds_strong, min_strong = self.read_foregrounds_jam(c_ex)
        background_weak = choose_background(self.backgrounds, None) # do not augment backgrounds
        background_strong = choose_background(self.backgrounds, None)

        # normalize backgrounds
        background_weak = normalize(background_weak, np.clip(np.random.normal(-30, 12), -50, min_weak + 5))
        background_strong = normalize(background_strong, np.clip(np.random.normal(-30, 12), -50, min_strong + 5))
        mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)

        if np.max(np.abs(mixture_weak)) > 1:
            mixture_weak = mixture_weak / np.max(np.abs(mixture_weak))

        mixture_strong = np.sum(np.stack([*foregrounds_strong, background_strong]), 0)
        if np.max(np.abs(mixture_strong)) > 1:
            mixture_strong = mixture_strong / np.max(np.abs(mixture_strong))

        target_frames = self.cfg["feats"]["max_len"]
        mixture_weak = get_feats(self.feats_func, mixture_weak, target_frames)
        mixture_strong = get_feats(self.feats_func, mixture_strong, target_frames)

        mixture_weak = apply_augmentation_chain(mixture_weak, self.cfg["augmentations"]["synth_unlabeled"]["weak_feats"])
        mixture_strong = apply_augmentation_chain(mixture_strong, self.cfg["augmentations"]["synth_unlabeled"]["strong_feats"])

        mixture_weak = torch.from_numpy(mixture_weak.T).float()
        mixture_strong = torch.from_numpy(mixture_strong.T).float()

        labels = c_ex["labels"]
        factor = (self.cfg["feats"]["hop_size"] / self.cfg["data"]["sample_rate"]) * self.cfg["net"]["pool_factor"]
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

        return mixture_weak, mixture_strong, strong, weak, mask_strong, mask_weak


class StrongSetSimpleLabeled(Dataset):
    """we use the foregrounds and backgrounds created with Scaper but dynamically mix them and augment them"""

    def __init__(self, jams, cfg, encoder, time_augment=True,
                  backgrounds=None, rirs=None):

        self.jams = jams
        self.encoder = encoder
        self.cfg = cfg
        self.time_augment = time_augment
        # we get the single sources and augment them singularly
        # we get all backgrounds
        self.feats_func = choose_feats(cfg["feats"])
        self.rirs = rirs
        self.backgrounds = backgrounds

    def __len__(self):
        return len(self.jams)

    def read_foregrounds_jam(self, c_ex):

        foregrounds = []
        for s in c_ex["foreground"]:
                tmp, fs = sf.read(s)
                if len(tmp.shape) > 1:
                    if self.time_augment:
                        tmp = tmp[:, np.random.randint(0, tmp.shape[-1]-1)]
                    else:
                        tmp = np.mean(tmp, -1)
                tmp = downsample(tmp, fs)
                tmp = tmp - np.mean(tmp) # zero mean
                foregrounds.append(tmp)

        # we have a list of audios
        if self.time_augment:
            # apply time domain augmentation to each foreground separately
            foregrounds_weak = []
            min_lvl_weak = np.inf
            first_weak = None

            orig_len = int(self.cfg["data"]["sample_rate"]*self.cfg["data"]["max_len_seconds"])

            for f in foregrounds:
                tmp = apply_augmentation_chain(f, self.cfg["augmentations"]["synth_labeled"]["weak_time"], self.rirs)
                if not first_weak:
                    c_lvl = np.clip(np.random.normal(-30, 7), -45, 0)
                else:
                    c_lvl = np.clip(np.random.normal(first_weak - 4, 3), max(first_weak - 20, -45), min(first_weak + 20, 0))
                tmp = normalize(tmp, c_lvl)
                min_lvl_weak = min(min_lvl_weak, c_lvl)
                tmp = np.pad(tmp, (0, orig_len-len(tmp)), mode="constant")
                foregrounds_weak.append(tmp)

            return foregrounds_weak, min_lvl_weak
        else:
            return foregrounds

    def get_labels(self, labels):

        return self.encoder.encode_strong_df(labels)

    def __getitem__(self, item):

        c_ex = self.jams[item]
        return self.example_from_jams(c_ex)

    def example_from_jams(self, c_ex):
        """ Useful for reusability by child classes"""
        # we only weak augment
        if self.time_augment:
            foregrounds_weak, min_weak = self.read_foregrounds_jam(c_ex)
            background_weak = choose_background(self.backgrounds, None)

            # normalize backgrounds
            background_weak = normalize(background_weak, np.clip(np.random.normal(-30, 12), -50, min_weak + 5))
            mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)
        else:
            foregrounds_weak = self.read_foregrounds_jam(c_ex)
            background_weak = choose_background(self.backgrounds, None)
            mixture_weak = np.sum(np.stack([*foregrounds_weak, background_weak]), 0)

        if np.max(np.abs(mixture_weak)) > 1:
            mixture_weak = mixture_weak / np.max(np.abs(mixture_weak))

        target_frames = self.cfg["feats"]["max_len"]
        mixture_weak = get_feats(self.feats_func, mixture_weak, target_frames)

        mixture_weak = apply_augmentation_chain(mixture_weak, self.cfg["augmentations"]["synth_labeled"]["weak_feats"])
        mixture_weak = torch.from_numpy(mixture_weak.T).float()
        mixture_strong = torch.zeros_like(mixture_weak)

        labels = c_ex["labels"]
        # to steps
        factor = (self.cfg["feats"]["hop_size"] / self.cfg["data"]["sample_rate"]) * self.cfg["net"]["pool_factor"]
        labels = [[z, int(x/factor), int(np.ceil(y/factor))] for z, x, y in labels]
        strong = self.get_labels(labels)
        weak = np.sum(strong, 0) >= 1
        weak = torch.from_numpy(weak).float()
        strong = torch.from_numpy(strong).float()

        mask_weak = torch.zeros([1]).bool()
        mask_strong = torch.ones([1]).bool()

        # mixture strong is dummy
        return mixture_weak, mixture_strong, strong, weak, mask_strong, mask_weak


class StrongSetMixLabeled(StrongSetSimpleLabeled):
    """ Split the jams in separate events. __getitem__ will return single event."""
    def __init__(self, jams, cfg, encoder, time_augment=True,
                 backgrounds=None, rirs=None, mixup_prob=0.):
        self.class_dict = {'Alarm_bell_ringing': [],
                           'Blender': [],
                           'Cat': [],
                           'Dishes': [],
                           'Dog': [],
                           'Electric_shaver_toothbrush': [],
                           'Frying': [],
                           'Running_water': [],
                           'Speech': [],
                           'Vacuum_cleaner': []}
        self.jams_dict = copy.deepcopy(self.class_dict)
        jams_splitted = self.split_all_jams(jams)  # Also populate class_dict & jams_dict with list of all events
        super().__init__(jams_splitted, cfg, encoder, time_augment=time_augment,
                         backgrounds=backgrounds, rirs=rirs)
        self.mixup_prob = mixup_prob

    def split_all_jams(self, jams_list):
        """ Single events from jams list"""
        new = []
        for j in jams_list:
            new += self.split_one_jams(j)
        return new

    def split_one_jams(self, jam):
        """ Split one jams into several jams with seperated events"""
        splited = []
        for f, l in zip(jam['foreground'], jam['labels']):
            local_jam = dict(background='', mixture='', foreground=[f], labels=[l])
            splited.append(local_jam)
            self.class_dict[l[0]].append(f)
            self.jams_dict[l[0]].append(local_jam)
        return splited

    def merge_jams(self, *jams):
        if len(jams) == 1:
            return jams[0]
        j = dict(jams[0])
        for jbis in jams[1:]:
            j['labels'] += jbis['labels']
            j['foreground'] += jbis['foreground']
        return j

    def __getitem__(self, item):
        # Mixup with a given proba
        do_mixup = np.random.choice([True, False], p=[self.mixup_prob, 1 - self.mixup_prob])
        if do_mixup:

            mixture_weak1, mixture_strong1, strong1, weak1, mask_strong1, mask_weak1 = self.create_one_ex()
            mixture_weak2, mixture_strong2, strong2, weak2, mask_strong2, mask_weak2 = self.create_one_ex()

            # Mix two generated examples
            alpha = np.random.uniform(0.05, 0.95)
            mixture_weak = alpha * mixture_weak1 + (1-alpha) * mixture_weak2
            mixture_strong = alpha * mixture_strong1 + (1-alpha) * mixture_strong2
            strong = alpha * strong1 + (1-alpha) * strong2
            weak = alpha * weak1 + (1-alpha) * weak2
            # Masks are dummy in parent, use the first one
            return mixture_weak, mixture_strong, strong, weak, mask_strong1, mask_weak1
        else:
            return self.create_one_ex()

    def create_one_ex(self):
        # -  # with 1 uniq events:  876  (33%)
        # -  # with 2 uniq events:  1226 (48%=
        # -  # with 3 uniq events:  436 (16%=
        # -  # with 4 uniq events:  43 (1.6%)
        prob = [0.3333333333333333, 0.4848484848484848, 0.1616161616161616, 0.0202020202020202]
        n_unique_src = np.random.choice([1, 2, 3, 4], p=prob)
        labels = list(np.random.choice(list(self.class_dict.keys()), n_unique_src))
        all_jams = []
        for l in labels:
            n_events = np.random.choice([1, 2, 3])
            j = list(np.random.choice(self.jams_dict[l], n_events))
            all_jams += j
        final_jams = self.merge_jams(*all_jams)
        return super().example_from_jams(final_jams)



"""
2584 Examples
- # with 1 uniq events:  876  (33%)
- # with 2 uniq events:  1226 (48%=
- # with 3 uniq events:  436 (16%=
- # with 4 uniq events:  43 (1.6%)
- # with 5 uniq events:  3 

3.5 events (non unique) in average

# Number of non-unique events per example
- 1 557
- 2 520
- 3 460 
- 4 348
- 5 261
- 6 154
- 7 118
- 8 77
- 9 38
- 10 22
"""
