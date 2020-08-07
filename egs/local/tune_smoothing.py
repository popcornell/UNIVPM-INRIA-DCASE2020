import yaml
import os
from librosa.sequence import viterbi_discriminative, transition_loop
import glob
from data_utils.parse_scaper import parse_validation
from data_utils.valset import ValSet
import numpy as np
from SED.baseline_tools.baseline import SEDBaseline
from SED.baseline_tools.utilities.ManyHotEncoder import ManyHotEncoder
from data_utils.label_hashtable import label_hashtable
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from SED.baseline_tools.evaluation_measures import compute_sed_eval_metrics
import scipy.ndimage

with open("/home/sam/Projects/DCASE2020/egs/fixmatch/exp/dat/confs.yml", "r") as f:
    confs = yaml.load(f)

# step 0 load dataset on which optimize ## we need strong labels

validation_files = glob.glob(os.path.join("/media/sam/bx500/DCASE_DATA/dataset/audio/validation_16k/", "*.wav"))
validation_meta = os.path.join("/media/sam/bx500/DCASE_DATA/dataset/metadata/", "validation", "validation.tsv")
validation = parse_validation(validation_files, validation_meta)

# we split validation for tuning
split_factor = int(0.5*len(validation))
np.random.shuffle(validation)
train = validation[:split_factor]
valid = validation[split_factor:]


# we may add synthetic training also
encoder = ManyHotEncoder(list(label_hashtable.keys()), confs["feats"]["max_len"] // confs["net"]["pool_factor"])

train = ValSet(train, confs, encoder, return_filename=True)
valid = ValSet(valid, confs, encoder, return_filename=True)

whole_valid = ValSet(validation, confs, encoder, return_filename=True)
# step 1 we load model we want to optimize


model = SEDBaseline(confs, load_pretrained=True, scaling="Scaler", use_pcen=False) # load pre trained (included scaler)
model.eval()
model = model.cuda()


# init hyperparameter serach space
p_sil = Real(low=0.01, high=0.999, name='p_sil')
p_events = [Real(low=0.01, high=0.999, name='p_{}'.format(k)) for k in label_hashtable.keys()]


def apply_HMMs(preds, transitions):

    decoded = []
    for i in range(preds.shape[-1]):
        c_log = preds[:, i]
        c_dec = viterbi_discriminative(np.vstack([1 - c_log, c_log]), transitions[i])
        decoded.append(c_dec)
    decoded = np.stack(decoded).T

    return decoded

def delete_shorter(preds, factor):
    def helper(segments, th=np.inf):

        tmp = []
        for s, e in segments:
            if (e - s) > th:
                tmp.append([s, e])
        return tmp

    preds_dict = {}
    for entry in preds:
        event_name = entry[0]
        if event_name not in preds_dict.keys():
            preds_dict[event_name] = [[entry[1], entry[2]]]
        else:
            preds_dict[event_name].append([entry[1], entry[2]])

    for k in preds_dict.keys():
        preds_dict[k] = helper(preds_dict[k], th=factor)

    out = []
    for k in preds_dict.keys():
        for segs in preds_dict[k]:
            out.append([k, segs[0], segs[1]])
    return out




def merge_shorter(preds, factor):
    def helper(intervals, delta=0.0):
        """
        A simple algorithm can be used:
        1. Sort the intervals in increasing order
        2. Push the first interval on the stack
        3. Iterate through intervals and for each one compare current interval
           with the top of the stack and:
           A. If current interval does not overlap, push on to stack
           B. If current interval does overlap, merge both intervals in to one
              and push on to stack
        4. At the end return stack
        """
        if not intervals:
            return intervals
        intervals = sorted(intervals, key=lambda x: x[0])

        merged = [intervals[0]]
        for current in intervals:
            previous = merged[-1]
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            else:
                merged.append(current)
        return merged

    preds_dict = {}
    for entry in preds:
        event_name = entry[0]
        if event_name not in preds_dict.keys():
            preds_dict[event_name] = [[entry[1], entry[2]]]
        else:
            preds_dict[event_name].append([entry[1], entry[2]])

    for k in preds_dict.keys():
        preds_dict[k] = helper(preds_dict[k], factor)

    out = []
    for k in preds_dict.keys():
        for segs in preds_dict[k]:
            out.append([k, segs[0], segs[1]])
    return out


def evaluate(model, train, encoder, transitions):

    pd_preds = pd.DataFrame()
    pd_truths = pd.DataFrame()
    # evaluate on train
    for batch in DataLoader(train, batch_size=8, drop_last=False):
        mixture, strong, weak, filenames = batch
        mixture = mixture.cuda()

        strong_pred, _, _ = model(mixture.unsqueeze(1))  # assume model also outpts logits

        for j in range(strong_pred.shape[0]):  # over batches
            pred = strong_pred[j].cpu().detach().numpy()

            # apply HMMs
            pred = apply_HMMs(pred, transitions)
            #pred = pred > 0.5
            #pred = scipy.ndimage.filters.median_filter(pred, (7, 1))
            pred = encoder.decode_strong(pred)
            # post processing on segments
            # merge shorter and delete shorter
            pred = delete_shorter(pred, 3)
            pred = merge_shorter(pred, 3)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"

            truth = encoder.decode_strong(strong[j].cpu().detach().numpy())
            truth = pd.DataFrame(truth, columns=["event_label", "onset", "offset"])
            truth["filename"] = Path(filenames[j]).stem + ".wav"
            pd_preds = pd_preds.append(pred)
            pd_truths = pd_truths.append(truth)

    time_pooling = confs["net"]["pool_factor"]
    sample_rate = confs["data"]["sample_rate"]
    hop_size = confs["feats"]["hop_size"]
    pd_preds.loc[:, "onset"] = pd_preds.onset * time_pooling / (
            sample_rate / hop_size)
    pd_preds.loc[:, "offset"] = pd_preds.offset * time_pooling / (
            sample_rate / hop_size)
    pd_preds = pd_preds.reset_index(drop=True)

    pd_truths.loc[:, "onset"] = pd_truths.onset * time_pooling / (
            sample_rate / hop_size)
    pd_truths.loc[:, "offset"] = pd_truths.offset * time_pooling / (
            sample_rate / hop_size)
    pd_truths = pd_truths.reset_index(drop=True)

    event_train, segment_train = compute_sed_eval_metrics(pd_preds, pd_truths)
    f_score_train = event_train.results()["class_wise_average"]['f_measure']['f_measure']

    return f_score_train


def get_transitions(probs):
    p_sil = probs[0]
    p_events = probs[1:]
    transitions = []
    for i in range(len(p_events)):  # one hmm for each class, p_sil is tied
        transitions.append(transition_loop(2, [p_sil, p_events[i]]))

    return transitions


def tuna(model, train, validation, encoder, probs):

    # init HMMs transition probs
    transitions = get_transitions(probs)

    f_score_train = evaluate(model, train, encoder, transitions)

    print("Train f1 {}".format(f_score_train))

    f_score_valid = evaluate(model, validation, encoder, transitions)

    print("Validation f1 {}".format(f_score_valid))
    print("hyperpars {}".format(probs))

    return 1. - f_score_train # maximize f_score on train

transitions = get_transitions([0.8936181121106661, 0.31550391085547114, 0.8923570974604905, 0.6490908582453855, 0.949481580049081, 0.7288818269203898, 0.4234812116634173, 0.5027625909657499, 0.8155335690232651, 0.6838988940188258, 0.6928248693985566])

print(evaluate(model, whole_valid, encoder, transitions))

helper = lambda x : tuna(model, train, valid,  encoder,  probs=x)
search_result = forest_minimize(helper, [p_sil, *p_events], n_calls=200)
print("#" * 100)
print("#" * 100)
print("BEST HYPERPARAMS on train: {}".format(search_result.x))