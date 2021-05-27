import ast
import gc
import os
import math
import random
import time
import warnings
import sys
sys.path.append("/root/workspace/BirdCLEF2021")

import albumentations as A
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import colorednoise as cn
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.models as models

from pathlib import Path
from typing import List

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from sklearn import model_selection
from sklearn import metrics
from timm.models.layers import SelectAdaptivePool2d
from torch.optim.optimizer import Optimizer
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation

from tqdm import tqdm

import albumentations as A
import audiomentations as AD

from apex import amp


class CFG:
    EXP_ID = '030' 

    ######################
    # Globals #
    ######################
    seed = 6718
    epochs = 30 # 80
    # cutmix_and_mixup_epochs = 75
    train = True
    folds = [1]
    img_size = 224
    main_metric = "epoch_f1_at_03"
    minimize_metric = False

    ######################
    # Data #
    ######################
    train_datadir = Path("inputs/train_short_audio")
    train_csv = "inputs/train_metadata.csv"
    train_soundscape = "inputs/train_soundscape_labels.csv"

    ######################
    # Dataset #
    ######################
    transforms = {
        "train": [{"name": "Normalize"}],
        "valid": [{"name": "Normalize"}]
    }
    period = 5
    n_mels = 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224,
        "fmin": 20,
        "fmax": 16000
    }

    target_columns = [
        'acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro',
        'amegfi', 'amekes', 'amepip', 'amered', 'amerob',
        'amewig', 'amtspa', 'andsol1', 'annhum', 'astfly',
        'azaspi1', 'babwar', 'baleag', 'balori', 'banana',
        'banswa', 'banwre1', 'barant1', 'barswa', 'batpig1',
        'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher',
        'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo',
        'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1',
        'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho',
        'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow', 'bobfly1',
        'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla',
        'brncre', 'brnjay', 'brnthr', 'brratt1', 'brwhaw', 'brwpar1',
        'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan',
        'buhvir', 'bulori', 'burwar1', 'bushti', 'butsal1', 'buwtea',
        'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar',
        'carchi', 'carwre', 'casfin', 'caskin', 'caster1', 'casvir',
        'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1',
        'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob',
        'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 'colcha1', 'coltro1',
        'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1',
        'comrav', 'comyel', 'coohaw', 'cotfly1', 'cowscj1', 'cregua1',
        'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly',
        'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro',
        'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa', 'gadwal',
        'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo',
        'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1',
        'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 'grcfly',
        'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel',
        'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra', 'grycat',
        'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr',
        'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa', 'houwre',
        'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo', 'larspa',
        'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1',
        'lesvio1', 'linspa', 'linwoo1', 'littin1', 'lobdow', 'lobgna5', 'logshr',
        'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre',
        'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1',
        'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar',
        'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit', 'obnthr1', 'ocbfly1',
        'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar',
        'osprey', 'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2',
        'phivir', 'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1',
        'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut',
        'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1',
        'rehbar1', 'relpar', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinkin1',
        'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum', 'rtlhum',
        'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar',
        'rucwar1', 'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1',
        'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 'scamac1',
        'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2',
        'sinwre1', 'sltred', 'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan',
        'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1', 'squcuc1', 'stbori',
        'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2',
        'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin',
        'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc', 'tropar',
        'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa', 'warvir',
        'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea',
        'westan', 'wewpew', 'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa',
        'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1',
        'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc',
        'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1',
        'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla', 'yehcar1', 'yelgro',
        'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir'] \
    + ['nocall']

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 64, 
            "num_workers": 0,
            "shuffle": True
        },
        "valid": {
            "batch_size": 128,
            "num_workers": 0,
            "shuffle": False
        }
    }

    ######################
    # Split #
    ######################
    split = "StratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 6718
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b0_ns"
    pooling = "max"
    pretrained = True
    num_classes = 398
    in_channels = 1

    N_FOLDS = 5
    LR = 1e-3
    apex = True
    T_max=10
    min_lr=1e-6


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 mode='train'):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        
        wav_path = sample["filepath"]
        start_sec = sample["start_seconds"]
        end_sec = sample["end_seconds"]
        len_label = sample["len_label"]
        labels = sample["primary_label"]
        secondary_labels = sample["secondary_labels"]

        y, sr = sf.read(wav_path)

        len_y = len(y)
        effective_length = sr * 5
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            if self.mode == 'train':
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            if self.mode == 'train':
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        y = np.nan_to_num(y)

        y = audio_augmenter(y)

        y = np.nan_to_num(y)

        all_targets = np.zeros(len(CFG.target_columns), dtype=float)

        targets = np.zeros(len(CFG.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[CFG.target_columns.index(ebird_code)] = 1.0
            all_targets[CFG.target_columns.index(ebird_code)] = 1.0

        secondary_targets = np.zeros(len(CFG.target_columns), dtype=float)
        if secondary_labels is not None:
             for ebird_code in secondary_labels.split():
                 if ebird_code == 'rocpig1':
                     ebird_code = 'rocpig'
                 secondary_targets[CFG.target_columns.index(ebird_code)] = 1.0
                 all_targets[CFG.target_columns.index(ebird_code)] = 1.0


        return {
            "image": y.reshape(1, -1),
            "all_targets": all_targets,
            "primary_targets": targets,
            "secondary_targets": secondary_targets
        }


def get_transforms(phase: str):
    transforms = CFG.transforms
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None
        
        
class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


# Mostly taken from https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="zero"):
        super().__init__(always_apply, p)

        assert padding_mode in [
            "replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second,
                                  self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        return augmented


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit = db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output



class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class PANNsDense121Att(nn.Module):
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int, classes_num: int, apply_aug: bool, top_db=None):
        super().__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        self.interpolate_ratio = 32  # Downsampled ratio
        self.apply_aug = apply_aug

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlockV2(1024, classes_num, activation='sigmoid')


        self.densenet_features = models.densenet121(pretrained=True).features

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def cnn_feature_extractor(self, x):
        x = self.densenet_features(x)
        return x
    
    def preprocess(self, input_x, mixup_lambda=None):

        x = self.spectrogram_extractor(input_x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.apply_aug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training  and self.apply_aug and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        return x, frames_num
        

    def forward(self, input_data):
        # input_x, mixup_lambda = input_data
        input_x = input_data
        mixup_lambda = None
        """
        Input: (batch_size, data_length)"""
        b, c, s = input_x.shape
        input_x = input_x.reshape(b*c, s)
        x, frames_num = self.preprocess(input_x, mixup_lambda=mixup_lambda)
        if mixup_lambda is not None:
            b = (b*c)//2
            c = 1
        # Output shape (batch size, channels, time, frequency)
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
        x = self.cnn_feature_extractor(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        frame_shape =  framewise_output.shape
        clip_shape = clipwise_output.shape
        output_dict = {
            'framewise_output': framewise_output.reshape(b, c, frame_shape[1],frame_shape[2]),
            'clipwise_output': clipwise_output.reshape(b, c, clip_shape[1]),
        }

        return output_dict


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss



EPSILON_FP16 = 1e-5


class SedScaledPosNegFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0, secondary_factor=1.0):
        super().__init__()

        self.loss_fn = nn.BCELoss(reduction='none')
        self.secondary_factor = secondary_factor
        self.gamma = gamma
        self.alpha_1 = alpha_1
        self.alpha_0 = alpha_0
        self.loss_keys = ["bce_loss", "F_loss", "FScaled_loss", "F_loss_0", "F_loss_1"]

    def forward(self, y_pred, y_target):
        y_true = y_target[0] # y_target["all_targets"]
        y_sec_true = y_target[1] # y_target["secondary_targets"]
        # bs, s, o = y_true.shape

        # Sigmoid has already been applied in the model
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        # y_pred = y_pred.reshape(bs*s,o)
        # y_true = y_true.reshape(bs*s,o)
        # y_sec_true = y_sec_true.reshape(bs*s,o)

        with torch.no_grad():
            y_all_ones_mask = torch.ones_like(y_true, requires_grad=False)
            y_all_zeros_mask = torch.zeros_like(y_true, requires_grad=False)
            y_all_mask = torch.where(y_true > 0.0, y_all_ones_mask, y_all_zeros_mask)
            y_ones_mask = torch.ones_like(y_sec_true, requires_grad=False)
            y_zeros_mask = torch.ones_like(y_sec_true, requires_grad=False) *self.secondary_factor
            y_secondary_mask = torch.where(y_sec_true > 0.0, y_zeros_mask, y_ones_mask)
        bce_loss = self.loss_fn(y_pred, y_true)
        pt = torch.exp(-bce_loss)
        F_loss_0 = (self.alpha_0*(1-y_all_mask)) * (1-pt)**self.gamma * bce_loss
        F_loss_1 = (self.alpha_1*y_all_mask) * (1-pt)**self.gamma * bce_loss

        F_loss = F_loss_0 + F_loss_1

        FScaled_loss = y_secondary_mask*F_loss
        FScaled_loss = FScaled_loss.mean()

        # return FScaled_loss, {"bce_loss": bce_loss.mean(), "F_loss_1": F_loss_1.mean(), "F_loss_0": F_loss_0.mean(), "F_loss": F_loss.mean(), "FScaled_loss": FScaled_loss }
        return FScaled_loss


# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        # self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred["clipwise_output"].cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.f1_03 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.3, average="samples")
        self.f1_05 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.5, average="samples")
        
        return {
            "f1_at_03" : self.f1_03,
            "f1_at_05" : self.f1_05,
        }
    
        
# def loss_fn(y_pred, y_all, y_second):
    # loss_fct = SedScaledPosNegFocalLoss()
    # loss = loss_fct(y_pred["clipwise_output"], (y_all, y_second))
def loss_fn(logits, targets):
    loss_fct = BCEFocalLoss() # BCEFocal2WayLoss()
    loss = loss_fct(logits['clipwise_output'], targets)
    return loss
        
        
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['all_targets'].to(device)
        secondary_targets = data['secondary_targets'].to(device)
        outputs = model(inputs)
        # loss = loss_fn(outputs, targets, secondary_targets)
        loss = loss_fn(outputs, targets)
        if CFG.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['image'].to(device)
            targets = data['targets'].to(device)
            secondary_targets = data['secondary_targets'].to(device)
            outputs = model(inputs)
            # loss = loss_fn(outputs, targets, secondary_targets)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


mean = (0.485, 0.456, 0.406) # RGB
std = (0.229, 0.224, 0.225) # RGB

albu_transforms = {
    'train' : A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Cutout(max_h_size=5*4, max_w_size=16*3, p=0.3),
            A.Normalize(mean, std),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ]),
}

audio_augmenter = Compose([
            OneOf([
                GaussianNoiseSNR(min_snr=10),
                PinkNoiseSNR(min_snr=10)
            ]),
            TimeShift(sr=32000),
            VolumeControl(p=0.5),
            Normalize()
])


model_config =  {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 398,
            "apply_aug": False, # True,
            "top_db": None
        }


OUTPUT_DIR = f'outputs/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=Path("logs") / f"train_{CFG.EXP_ID}.log")

# environment
set_seed(CFG.seed)
device = get_device()

# data
# train = pd.read_csv('inputs/image_folds.csv')
# train['filepath'] = train['filepath'].map(lambda x: 'inputs/train_images/' + '/'.join(x.split('/')[4:]))
train = pd.read_csv('inputs/folds.csv')
train['filepath'] = train['filepath'].map(lambda x: 'inputs/' + '/'.join(x.split('/')[3:]))

short_audio = train.loc[:62873].copy()
long_audio = train.loc[62874:].copy()

meta = pd.read_csv('inputs/train_metadata.csv')

short_audio['secondary_labels'] = meta['secondary_labels'].copy()
short_audio['secondary_labels'] = short_audio['secondary_labels'].map(lambda x: ' '.join(ast.literal_eval(x)))
long_audio['secondary_labels'] = None

short_audio['rating'] = meta['rating'].copy()
long_audio['rating'] = 999 # -1


new_train = pd.concat([short_audio, long_audio]).reset_index(drop=True)

# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 120)

    trn_df = new_train[new_train['kfold']!=fold].reset_index(drop=True)
    val_df = new_train[new_train.kfold == fold].reset_index(drop=True)

    loaders = {
        phase: torchdata.DataLoader(
            WaveformDataset(
                df_,
                mode=phase
            ),
            **CFG.loader_params[phase])  # type: ignore
        for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
    }

    model = PANNsDense121Att(**model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    model = model.to(device)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)


    p = 0
    min_loss = 999
    best_score = -np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        # if epoch < CFG.cutmix_and_mixup_epochs:
        #     train_avg, train_loss = train_mixup_cutmix_fn(model, loaders['train'], device, optimizer, scheduler)
        # else: 
        #     train_avg, train_loss = train_fn(model, loaders['train'], device, optimizer, scheduler)

        train_avg, train_loss = train_fn(model, loaders['train'], device, optimizer, scheduler)
        valid_avg, valid_loss = valid_fn(model, loaders['valid'], device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}")
        logger.info(f"Epoch {epoch+1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}")

        if valid_avg['f1_at_03'] > best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_03']}")
            logger.info(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['f1_at_03']

