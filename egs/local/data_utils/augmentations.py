import random
import numpy as np
from scipy.signal import fftconvolve, lfilter
from pysndfx import AudioEffectsChain
import soundfile as sf
import scipy.ndimage
from .spec_augment.specaugment import spec_augment
from pathlib import Path


class Augmentations:

    @staticmethod
    def noiseBurst(signal, magnitude):

        magnitude = eval(magnitude)

        L = signal.__len__()
        RMS = np.sqrt(np.mean(signal ** 2))
        # magnitude controls length and power of the additive noise
        awgn_rms = RMS * magnitude
        awgn_len = int(np.floor(L * magnitude))
        noise = awgn_rms * np.random.rand(awgn_len)
        ti = int(np.floor(np.random.random() * (L - awgn_len)))
        signal[ti:ti + awgn_len] = signal[ti:ti + awgn_len] + noise

        return signal

    @staticmethod
    def sineBurst(signal, magnitude, Fs=16000):

        magnitude = eval(magnitude)

        L = signal.__len__()
        RMS = np.sqrt(np.mean(signal ** 2))
        # magnitude controls length and power of the additive noise
        sine_rms = RMS * magnitude
        sine_len = int(np.floor(L * magnitude))
        # gen tone
        f0 = np.random.random() * 4000  # added sine has random f0 in range [0,4000]
        t = np.linspace(0, sine_len / Fs, sine_len)
        sine = np.sin(2 * np.pi * f0 * t)  # test tone
        sine = sine_rms * sine
        ti = int(np.floor(np.random.random() * (L - sine_len)))
        signal[ti:ti + sine_len] = signal[ti:ti + sine_len] + sine

        return signal

    @staticmethod
    def speed(signal, magnitude):

        magnitude = eval(magnitude)

        fx = (AudioEffectsChain().speed(magnitude))
        signal = fx(signal)
        return signal

    @staticmethod
    def bandpass(signal, center, width):

        center = eval(center)
        width = eval(width)

        fx = (AudioEffectsChain().bandpass(center, width))
        signal = fx(signal)
        return signal

    @staticmethod
    def tremolo(signal, freq, depth):

        freq = eval(freq)
        depth = eval(depth)

        fx = (AudioEffectsChain().tremolo(freq, depth))
        signal = fx(signal)
        return signal

    @staticmethod
    def bend(signal, n_bends, delay, cents, duration):
        raise NotImplementedError
        n_bends = eval(n_bends)
        bends = [[eval(delay), eval(cents), eval(duration)] for x in range(n_bends)]
        fx = (AudioEffectsChain().bend(bends))
        signal = fx(signal)
        return signal

    @staticmethod
    def chorus(signal, gain_in, gain_out, n_decays, delay, decay, speed, depth):
        raise NotImplementedError

        fx = (AudioEffectsChain().chorus(gain_in, gain_out, decays))
        signal = fx(signal)
        return signal

    @staticmethod
    def contrast(signal, magnitude):

        magnitude = eval(magnitude)

        fx = (AudioEffectsChain().custom("contrast {}".format(magnitude)))
        signal = fx(signal)
        return signal

    @staticmethod
    def hilbert(signal, magnitude):

        magnitude = eval(magnitude)
        if magnitude % 2 == 0:
            magnitude+= 1

        fx = (AudioEffectsChain().custom("hilbert -n {}".format(magnitude)))
        signal = fx(signal)
        return signal

    @staticmethod
    def overdrive(signal, gain, colour):

        gain = eval(gain)
        colour = eval(colour)

        fx = (AudioEffectsChain().overdrive(gain, colour))
        signal = fx(signal)
        return signal

    @staticmethod
    def pitch(signal, shift):

        shift = eval(shift)

        fx = (AudioEffectsChain().pitch(shift))
        signal = fx(signal)
        return signal

    @staticmethod
    def tempo(signal, factor):

        factor = eval(factor)

        fx = (AudioEffectsChain().tempo(factor))
        signal = fx(signal)
        return signal

    @staticmethod
    def highshelf(signal, gain, frequency):

        gain = eval(gain)
        frequency = eval(frequency)

        fx = (AudioEffectsChain().highshelf(gain, frequency))
        signal = fx(signal)
        return signal

    @staticmethod
    def lowshelf(signal, gain, frequency):

        gain = eval(gain)
        frequency = eval(frequency)

        fx = (AudioEffectsChain().lowshelf(gain, frequency))
        signal = fx(signal)
        return signal

    @staticmethod
    def fade(signal, magnitude):

        magnitude = eval(magnitude)

        L = signal.__len__()
        #RMS = np.sqrt(np.mean(signal ** 2))

        # type can be fade in/out
        ftype = np.random.random() < 0.5
        dur = int(np.floor(L * magnitude))
        minlev = 1 - magnitude
        if ftype == False:  # fade in
            gain = np.linspace(minlev ** 3, 1, dur) # exponentiating is a trick to unlinearize
            signal[0:dur] = signal[0:dur] * gain
        else:  # fade out
            gain = np.linspace(1, minlev ** 3, dur)
            signal[L - dur:] = signal[L - dur:] * gain

        return signal

    @staticmethod
    def roll(signal, magnitude):
        magnitude = eval(magnitude)
        return np.roll(signal, magnitude)

    @staticmethod
    def echoes(signal, magnitude, Fs=16000):

        magnitude = eval(magnitude)

        # multitap comb
        CL = int(np.floor(0.5 * Fs))  # max 500ms
        dly = np.zeros(CL)
        dly[0] = 1
        for ntaps in range(5):
            i = int(np.floor(np.random.random() * CL))
            dly[i] = np.random.randn() * magnitude ** 3  # random gains

        signal = lfilter(dly, 1, signal)
        return signal

    @staticmethod
    def phasing(signal, magnitude, Fs=16000):

        magnitude = eval(magnitude)

        L = len(signal)

        # tv comb
        const = int(np.floor(0.1 * Fs))  # 100 ms
        maxvar = int(np.floor(1.9 * const))
        dly = np.zeros(L + const + maxvar)
        dly[0:L] = signal
        t = np.linspace(0, L / Fs, L)
        f = 0.5  # Hz
        var = (np.sin(2 * np.pi * f * t) + 1) * 0.9 * const
        y = np.zeros(L)
        for n in range(L):
            inte = np.floor(var[n])
            frac = var[n] - inte
            spill = (1 - frac) * dly[int(n + const + inte - 1)] + (frac) * dly[int(n + const + inte)]  # anticausal
            y[n] = signal[n] + magnitude ** 2 * spill

        return y #[0:L]

    @staticmethod
    def reverberate(signal, rirs, wet="1"):
        wet = eval(wet)

        rir = np.random.choice(rirs, 1)[0]

        if Path(rir).suffix == ".wav":
            rir, fs = sf.read(rir)
        elif Path(rir).suffix == ".rir.gz": # sunit rirs
            rir = np.genfromtxt(rir)
        else:
            raise NotImplementedError

        if len(rir.shape) > 1:
            rir = rir[:, np.random.randint(0, rir.shape[-1]- 1)]

        rev = fftconvolve(signal, rir)
        if wet == 1:
            return rev
        else:
            # pad to same shape
            signal = np.pad(signal, (0, len(rev)- len(signal)),  mode="constant")
            return wet*rev + (1-wet)*signal

    @staticmethod
    def gaussblur(image, sigma):

        sigma = eval(sigma)

        return scipy.ndimage.gaussian_filter(image, sigma)

    @staticmethod
    def spline(image, dummy):
        return scipy.ndimage.spline_filter(image, 3)

    @staticmethod
    def uniform(image, size):

        size = eval(size)

        return scipy.ndimage.uniform_filter(image, size)

    @staticmethod
    def median_filter(image, size):

        size = eval(size)

        return scipy.ndimage.median_filter(image, size)

    @staticmethod
    def minimum(image, size):

        size = eval(size)

        return scipy.ndimage.minimum(image, size)

    @staticmethod
    def maximum(image, size):

        size = eval(size)

        return scipy.ndimage.maximum(image, size)

    @staticmethod
    def rank(image, rank):

        rank = eval(rank)

        return scipy.ndimage.rank_filter(image, 1, size=rank)

    @staticmethod
    def percentile(image, percentile):

        percentile = eval(percentile)

        return scipy.ndimage.percentile_filter(image, percentile)

    @staticmethod
    def gaussnoise(feature, mean, snr):

        mean = eval(mean)
        snr = eval(snr)

        feature = feature[..., None]
        row, col, ch = feature.shape
        sigma = np.std(feature) / snr
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = feature + gauss
        return noisy[..., 0]

    @staticmethod
    def saltpepper(image, s_vs_p, amount):

        s_vs_p = eval(s_vs_p)
        amount = eval(amount)

        image = image[..., None]
        row, col, ch = image.shape

        image = np.copy(image)

        ceil = np.max(image)

        num_salt = np.ceil(amount * image.size * s_vs_p)
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))

        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in [row, col]]
        image[coords[0], coords[1], :] = np.random.uniform(ceil - (ceil // 2), ceil)

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in [row, col]]
        image[coords[0], coords[1], :] = 0

        return image[..., 0]

    @staticmethod
    def speckle(image, snr):

        snr = eval(snr)

        image = image[..., None]
        variance = np.std(image)**2 / snr
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * variance
        return noisy[..., 0]

    @staticmethod
    def poisson(image, dummy):
        noisy = np.random.poisson(image)
        return noisy[..., 0]

    @staticmethod
    def specaugment(image, frequency_masking_para,
                 time_masking_para, frequency_mask_num, time_mask_num):

        noisy = spec_augment(image, frequency_masking_para=frequency_masking_para,
                 time_masking_para=time_masking_para, frequency_mask_num=frequency_mask_num, time_mask_num=time_mask_num)
        return noisy


def apply_augmentation_chain(signal, confs,  rirs=None):

    # we first sample k augmentations from all possible ones
    if not confs["max"]:
        return signal

    n_augm = random.randint(0, confs["max"]) # can be also no augmentation
    c_augm = np.random.choice(confs["list"], n_augm, replace=False) # order is random

    if len(signal.shape) == 1:
        orig_len = len(signal)

    else:
        orig_len = signal.shape[-1]
    # we always reverberate but not on weak and unlabelled
    if rirs:
        signal = Augmentations.reverberate(signal, rirs, **confs["reverb"])

    for augm in c_augm:
        kwargs = confs[augm]
        signal = getattr(Augmentations, augm)(np.copy(signal), **kwargs)

    signal =  signal[..., :orig_len]
    if len(signal) < orig_len and len(signal.shape) == 1:
        signal = np.pad(signal, (0, orig_len - len(signal)), mode="constant")


    return signal
