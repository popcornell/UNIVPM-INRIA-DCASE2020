from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from .label_hashtable import label_hashtable
import glob

""" here in these functions we parse data created with Scaper """

def parse_jams(jams):
    out = []
    for j in jams:
        with open(j, "r") as f:
            tmp = json.load(f)
        mixwav = os.path.join(Path(j).parent, Path(j).stem) + ".wav"
        current = {"mixture": mixwav, "foreground": [], "background": [], "labels": []}

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

                current["foreground"].append(path)
                current["labels"].append([label, start, stop])
            else:
                raise EnvironmentError
        out.append(current)
    return out


def parse_weak(files, weak_meta):
    with open(weak_meta, "r") as f:
        label_meta = pd.read_csv(f, delimiter="\t")

    hashtable = {}
    for row in range(len(label_meta["filename"])):
        key = label_meta["filename"][row].strip(".wav")
        labels = label_meta["event_labels"][row].split(",")

        tmp = []
        for l in labels:
            tmp.append([l, np.nan, np.nan])
        hashtable[key] =  {"labels": tmp}

        # labels parsed now we parse audio files

    hashtable_files = {}
    for f in files:
        filename = Path(f).stem
        hashtable_files[filename] = {"filename": f}

    new = []
    for k in hashtable_files.keys():
        mixture = hashtable_files[k]["filename"]
        label = hashtable[k]["labels"]
        new.append({"mixture": mixture, "labels": label})

    return new


def parse_validation(files, meta):
    with open(meta, "r") as f:
        synth_meta = pd.read_csv(f, delimiter="\t")

    hashtable = {}
    for row in range(len(synth_meta["filename"])):
        key = synth_meta["filename"][row].strip(".wav")
        label = synth_meta["event_label"][row]
        onset = synth_meta["onset"][row]
        offset = synth_meta["offset"][row]
        if key not in hashtable.keys():
            hashtable[key] = [[label , onset, offset]]
        else:
            hashtable[key].append([label , onset, offset])

    # use full path as key
    hashtab_files = {}
    # hashtable with
    for f in files:
        filename = Path(f).stem
        if filename not in hashtab_files.keys():
            hashtab_files[filename] = {"filename": f}
        else:
            raise EnvironmentError

    # we merge now the two dicts with a mixture key and a label key
    new = []
    for k in hashtab_files.keys():
        mixture = hashtab_files[k]
        try:
            labels = hashtable[k]
        except:
            print("label missing")
        new.append({"mixture": mixture["filename"], "labels": labels})

    return new



