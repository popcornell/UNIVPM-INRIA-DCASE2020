import soundfile as sf
from pysndfx import AudioEffectsChain
from .augmentations import apply_augmentation_chain
import torch
import librosa
import numpy as np

def downsample(signal, fs, target=16000):

    if fs != target:
        return librosa.resample(signal, fs, target).astype("float32")
    else:
        return signal


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.
    The sequence should be on axis -2.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    shape = x.shape
    if shape[-2] <= max_len:
        padded = max_len - shape[-2]
        padded_shape = ((0, 0),)*len(shape[:-2]) + ((0, padded), (0, 0))
        x = np.pad(x, padded_shape, mode="constant")
    else:
        x = x[..., :max_len, :]
    return x


def choose_background(backgrounds_list, augm_hyperparams=None, target_len=None):

    c_background = np.random.choice(backgrounds_list, 1)[0]
    background, fs = sf.read(c_background)
    if len(background.shape) > 1:
        background = background[:, np.random.randint(0, background.shape[-1] -1)]
    background = downsample(background, fs)
    background = background - np.mean(background)

    # augment time domain
    if augm_hyperparams:
        background = apply_augmentation_chain(background, augm_hyperparams, rirs=None)

    if target_len:

        # we either pad or take random window to return the background
        if len(background) < target_len:
            background = np.pad(background, (0, target_len - len(background)), mode="constant")
        elif len(background) > target_len:
            offset = np.random.randint(0, len(background)-target_len)
            background = background[offset: offset + target_len]
        else:
            pass

    return background


def normalize(signal, target_dB):
    fx = (AudioEffectsChain().custom(
        "norm {}".format(target_dB)))
    signal = fx(signal)
    return signal



def amp_to_db(x):
    return torch.clamp(20*torch.log10(torch.clamp(x, min=1e-5)), max=80)

def choose_feats(configs):
    if configs["type"] == "mel_librosa":

        def libmel(x): # helper func
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()[0]

            x = x - np.mean(x)

            ham_win = np.hamming(configs["mel_librosa"]["n_fft"])

            x = librosa.stft(
                x,
                window=ham_win,
                center=True,
                pad_mode='reflect',
                n_fft=configs["mel_librosa"]["n_fft"],
                hop_length=configs["hop_size"]

            )

            x = librosa.feature.melspectrogram(
                S=np.abs(x),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
                htk=False, norm=None,
                n_mels=configs["mel_librosa"]["n_mels"], fmin=configs["mel_librosa"]["fmin"],
                fmax=configs["mel_librosa"]["fmax"], sr=configs["sr"])

            if configs["mel_librosa"]["take_log"]:
                x = librosa.amplitude_to_db(x)

            return x

        return lambda x: libmel(x).astype("float32")
    else:
        raise NotImplementedError


def get_feats(feats_func, x, target_frames):

    x = feats_func(x)
    x = pad_trunc_seq(x.T, target_frames).T

    return x