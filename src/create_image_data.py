import sys
sys.path.append("/root/workspace/BirdCLEF2021")

import os
import warnings
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


SEED = 6718
AUDIO_PATH = 'inputs/train_short_audio'
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NUM_WORKERS = 4
CLASSES = sorted(os.listdir(AUDIO_PATH) + ['nocall'])
NUM_CLASSES = len(CLASSES)
class AudioParams:
    """
    Parameters used for the audio data
    """
    sr = 32000
    # duration = 7

    # Melspectrogram
    n_mels = 128
    fmin = 20
    fmax = 16000
train = pd.read_csv('inputs/train_metadata.csv')
train["file_path"] = AUDIO_PATH + '/' + train['primary_label'] + '/' + train['filename']
paths = train["file_path"].values


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):
    """
    Normalizes an array in [0, 255] to the format adapted to neural network
    Arguments:
        image {np array [3 x H x W]} -- [description]
    Keyword Arguments:
        mean {None or np array} -- Mean for normalization, expected of size 3 (default: {None})
        std {None or np array} -- Std for normalization, expected of size 3 (default: {None})
    Returns:
        np array [H x W x 3] -- Normalized array
    """
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly, else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                    np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start: start + length]

    return y.astype(np.float32)


def get_wav_transforms():
    """
    Returns the transformation to apply on waveforms
    Returns:
        Audiomentations transform -- Transforms
    """
    transforms = Compose(
        [
            AddGaussianSNR(max_SNR=0.5, p=0.5),
        ]
    )
    return transforms

def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec

def load_audio(path, params):
    clip, _ = librosa.load(path, sr=params.sr, mono=True, res_type="kaiser_fast")
    return clip

def AtoI(path, params):
    y, sr = sf.read(path)
    # y = crop_or_pad(y, params.duration * params.sr, sr=params.sr, train=True, probs=None)
    
    y = y[sr:-sr] # Cut first 1sec and last 1sec, there may not be bird call.
    
    melspec = compute_melspec(y, params)
    image = mono_to_color(melspec)
    # image = normalize(image, mean=None, std=None)
    image = image.astype(np.uint8)
    return image

def save_(path):
    save_path = "inputs/train_images/" + "/".join(path.split('/')[-2:])
    np.save(save_path, AtoI(path, AudioParams))


for dir_ in CLASSES:
    _ = os.makedirs(f"inputs/train_images/{dir_}", exist_ok=True)
_ = Parallel(n_jobs=NUM_WORKERS)(delayed(save_)(AUDIO_PATH) for AUDIO_PATH in tqdm(paths))


