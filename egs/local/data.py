import glob
import json
import os
from pathlib import Path
from .data_utils.parse_scaper import parse_jams, parse_validation, parse_weak
from .data_utils.label_hashtable import label_hashtable


def parse_jams_sources(jams):
    out = []
    for j in jams:
        with open(j, "r") as f:
            tmp = json.load(f)
        mixwav = os.path.join(Path(j).parent, Path(j).stem) + ".wav"
        current = {"mixture": mixwav, "foreground": [], "background": []}

        for i, event in enumerate(tmp["annotations"][0]["sandbox"]["scaper"]["isolated_events_audio_path"]):
            if Path(event).stem.startswith("background"):
                path = os.path.join(Path(j).parents[5], event[3:])
                current["background"].append(path)
            elif Path(event).stem.startswith("foreground"):
                path = os.path.join(Path(j).parents[5], event[3:])

                start = tmp["annotations"][0]["data"][i]["value"]["event_time"]
                stop = start + tmp["annotations"][0]["data"][i]["value"]["event_duration"]
                label = tmp["annotations"][0]["data"][i]["value"]["label"]
                if label not in label_hashtable.keys():
                    if label.startswith("Frying"):
                        label = "Frying"
                    elif label.startswith("Vacuum_cleaner"):
                        label = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                current["foreground"].append([path, [label, start, stop]])
            else:
                raise EnvironmentError
        out.append(current)
    return out

def data_init(cfg, dynamic_jams=False):
    all_files = ['train_synth.json',
                 'weak.json',
                 'unlabeled.json',
                 'unlabeled_others.json',
                 'rirs.json',
                 'backgrounds.json',
                 'validation.json']
    all_path = ['cached/' + f for f in all_files]
    os.makedirs('cached/', exist_ok=True)
    if all(os.path.exists(p) for p in all_path):
        to_return = [
            json.load(open(p, 'r')) for p in all_path
        ]
        return to_return

    jams = glob.glob(os.path.join(cfg["data"]["synth_train_wav_folder"], "*.jams"))

    if not dynamic_jams:
        jams = parse_jams(jams)
        train_synth = jams
    else:
        train_synth = []
        tmp = parse_jams_sources(jams)
        for entry in tmp:
            train_synth.extend(entry["foreground"])


    weak_wavs = glob.glob(os.path.join(cfg["data"]["weak_wav_folder"], "*.wav"))
    weak_meta = os.path.join(cfg["data"]["metadata_root"], "train", "weak.tsv")
    weak = parse_weak(weak_wavs, weak_meta)


    validation_files = glob.glob(os.path.join(cfg["data"]["validation_wav_folder"], "*.wav"))
    validation_meta = os.path.join(cfg["data"]["metadata_root"], "validation", "validation.tsv")
    validation = parse_validation(validation_files, validation_meta)

    unlabeled = glob.glob(os.path.join(cfg["data"]["unlabeled_wav_folder"], "*.wav"))

    # we also add another voice --> unlabeled other basically we build a list of all weak plus all sources from synth_train_wav_folder

    unlabeled_others = glob.glob(os.path.join(cfg["data"]["synth_train_wav_folder"], "**/*foreground*.wav"), recursive=True)
    # backgrounds
    backgrounds = glob.glob(os.path.join(cfg["data"]["synth_train_wav_folder"], "**/*background*.wav"), recursive=True)
    #glob.glob(os.path.join(cfg["data"]["backgrounds_wav_root"], "**/*.wav"), recursive=True)

    # rirs
    rirs = []
    for f in cfg["data"]["rir_folders"]:
        rirs.extend(glob.glob(f, recursive=True))

    #TODO
    # not sure if we can add fuss sources and background to unlabeled data

    #fuss_sources = glob.glob(os.path.join(cfg["data"]["fuss_root"], "**/*foreground*.wav"), recursive=True)
    #fuss_backgrounds = glob.glob(os.path.join(cfg["data"]["fuss_root"], "**/*background*.wav"), recursive=True)
    # CACHE
    [
        json.dump(to_dump, open(path, 'w')) for to_dump, path in zip(
        [train_synth, weak, unlabeled, unlabeled_others, rirs, backgrounds, validation], all_path
    )
    ]
    return train_synth, weak, unlabeled, unlabeled_others, rirs, backgrounds, validation #, fuss_sources, fuss_backgrounds


if __name__ == "__main__":
    import yaml
    with open("/home/sam/Projects/DCASE2020/egs/fixmatch/conf/fixmatch.yml", "r") as f:
        confs = yaml.load(f)
    train_synth, weak, unlabeled, unlabeled_others, rirs, backgrounds, validation = data_init(confs)

    from data_utils.strongsetsimple import StrongSetSimpleUnlabeled, StrongSetSimpleLabeled
    from data_utils.weakset import WeakSetUnlabeled, WeakSetLabeled
    from data_utils.unlabeledset import UnlabeledSet
    from data_utils.multidataset import MultiDataset
    from data_utils.valset import ValSet
    from data_utils.concatdataset import ConcatDataset, MultiStreamBatchSampler


    from SED.baseline_tools.utilities.ManyHotEncoder import ManyHotEncoder
    from data_utils.label_hashtable import label_hashtable
    from torch.utils.data import DataLoader

    encoder = ManyHotEncoder(list(label_hashtable.keys()), confs["feats"]["max_len"] // confs["net"]["pool_factor"])

    valset = ValSet(validation, confs, encoder)


    synth_labeled = StrongSetSimpleLabeled(train_synth, confs, encoder=encoder, time_augment=True,
                                  backgrounds=backgrounds, rirs=rirs)



    synth_unlabeled = StrongSetSimpleUnlabeled(train_synth, confs, encoder, time_augment=True,
                                               backgrounds=backgrounds, rirs=rirs)

    c = ConcatDataset([synth_labeled, synth_unlabeled])
    m = MultiStreamBatchSampler(c, [8, 8])

    #for i in m:
        #print(i)

    weak_labeled_data = WeakSetLabeled(weak, confs, encoder, True)
    weak_unlabeled = WeakSetUnlabeled(weak, confs, encoder, backgrounds=backgrounds, rirs=rirs)

    for i in weak_unlabeled:
        print(i)

    unlabeled_in_domain = UnlabeledSet(unlabeled, confs, backgrounds=backgrounds, rirs=rirs) #WeakSetLabeled(weak, confs, encoder = encoder, time_augment=True)

    #for i in DataLoader(unlabeled_in_domain, batch_size=1, shuffle=True):
        #print(len(i))

    unlabeled_others = UnlabeledSet(unlabeled_others, confs, backgrounds=backgrounds, rirs=rirs)

    unlabeled_tot = MultiDataset([unlabeled_in_domain, synth_unlabeled, weak_unlabeled, unlabeled_others], [0.75, 0.15, 0.08, 0.02])

    print(len(unlabeled_tot))

    tot_data = MultiDataset([unlabeled_tot, weak_labeled_data, synth_labeled], [0.85, 0.1, 0.05])

    for i in DataLoader(tot_data, shuffle=True, batch_size=8):
        print(i)