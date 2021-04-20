# ====================================================
# Directory settings
# ====================================================
import os
import sys
import torch
sys.path.append("/root/workspace/BirdCLEF2021")


class CFG:
    EXP_ID = '001'
    SEED = 6718
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_FOLDS = 5
    IMG_DIR = 'inputs'
    TRAIN_PATH = 'inputs/train.csv'
    TEST_PATH = 'inputs/test.csv'
    SUB_PATH = 'inputs/sample_submission.csv'
    TRAIN_IMG_PATH = 'inputs/train_images/'
    TEST_IMG_PATH = 'inputs/test_images/'
    TRAIN_BATCH_SIZE = 8 # 8 * 2
    VALID_BATCH_SIZE = 16 # 16 * 2
    LR = 1e-3
    PATIENCE = 5
    DEBUG = False
    EPOCHS = 20 if not DEBUG else 3
    IMG_SIZE = 512
    MODEL_NAME = "tf_efficientnet_b1_ns"
    apex = True


OUTPUT_DIR = f'outputs/exp_{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


import copy
import gc
import time
import logging
import math
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam, SGD
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup

import albumentations as A
import albumentations.pytorch.transforms as T

import warnings
warnings.filterwarnings('ignore')

import timm

from tqdm import tqdm
tqdm.pandas()

if CFG.apex:
    from apex import amp

from src.machine_learning_util import trace, seed_everything, to_pickle, unpickle, setup_logger

sub_df = pd.read_csv('inputs/sample_submission.csv')
label_cols = list(sub_df.columns[1:])

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
LOGGER_PATH = OUTPUT_DIR+"log.txt" # f"logs/log_{CFG.EXP_ID}.txt"
setup_logger(CFG.SEED, LOGGER, FORMATTER, out_file=LOGGER_PATH)
LOGGER.info("seed={}".format(CFG.SEED))


mean = (0.485, 0.456, 0.406) # RGB
std = (0.229, 0.224, 0.225) # RGB

transform = {
    'train' : A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(int(CFG.IMG_SIZE * 1.5), int(CFG.IMG_SIZE * 1.5)),
        A.CenterCrop(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        A.Normalize(mean, std),
        T.ToTensorV2()
    ]),
    'val' : A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean, std),
        T.ToTensorV2()
    ]),    
}


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
        self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.score = roc_auc_score(np.array(self.y_true), np.array(self.y_pred))
        return {
            "AUC" : self.score,
        }
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def calculate_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.calculate_euclidean(anchor, positive)
        distance_negative = self.calculate_euclidean(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean() if size_average else losses.sum()        
        

def train_fn(model, data_loader, device, criterion, optimizer, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        anchor = data['anchor'].to(device)
        positive = data['pos'].to(device)
        negative = data['neg'].to(device)
        anc_embedding = model(anchor)
        pos_embedding = model(positive)
        neg_embedding = model(negative)
        loss = criterion(anc_embedding, pos_embedding, neg_embedding)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.update(loss.item(), anchor.size(0))
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def valid_fn(model, data_loader, device, criterion):
    model.eval()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk0:
            anchor = data['anchor'].to(device)
            positive = data['pos'].to(device)
            negative = data['neg'].to(device)
            anc_embedding = model(anchor)
            pos_embedding = model(positive)
            neg_embedding = model(negative)
            loss = criterion(anc_embedding, pos_embedding, neg_embedding)
            losses.update(loss.item(), anchor.size(0))
            tk0.set_postfix(loss=losses.avg)
    return losses.avg


def inference_fn(model, data_loader, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    anchors = []
    for data in tk0:
        anchor = data['anchor'].to(device)

        with torch.no_grad():
            anc_embedding = model(anchor).detach().cpu().numpy()
        
        anchors.append(anc_embedding.reshape(anc_embedding.shape[0], -1))
        
    anchors = np.concatenate(anchors)
    return anchors        


class TripletDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_path, transforms=None):
        self.label_group_id = df["label_group"].unique()
        self.df = df
        self.image_path = image_path
        self.transforms = transforms

    def __len__(self):
        return len(self.label_group_id)

    def __getitem__(self, idx: int):
        label_group_sample_id = self.label_group_id[idx]
        sample = self.df.query(f"label_group == {label_group_sample_id}")["image"]
        # 負例のサンプリングを行う
        while True:
            neg_sample_id = np.random.choice(self.label_group_id)
            if neg_sample_id != label_group_sample_id:
                break
        neg_sample = self.df.query(f"label_group == {neg_sample_id}")["image"]

        neg = np.random.choice(neg_sample)

        if len(sample) == 2:
            anchor = sample.iloc[0]
            pos = sample.iloc[-1]
        else:
            anchor = sample.iloc[0]
            pos = np.random.choice(sample.iloc[1:])
            
        anchor_img = cv2.imread(self.image_path+anchor)
        pos_img = cv2.imread(self.image_path+pos)
        neg_img = cv2.imread(self.image_path+neg)

        anchor = self.transforms(image=anchor_img)["image"]
        pos = self.transforms(image=pos_img)["image"]
        neg = self.transforms(image=neg_img)["image"]
        return {
            'anchor': torch.FloatTensor(anchor), 
            'pos': torch.FloatTensor(pos), 
            'neg': torch.FloatTensor(neg)
        }


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_path, transforms=None):
        self.df = df
        self.image_path = image_path
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_id = self.df['image'].values[idx]
            
        anchor_img = cv2.imread(self.image_path+img_id)

        anchor = self.transforms(image=anchor_img)["image"]

        return {
            'anchor': torch.FloatTensor(anchor), 
        }


class EffnNet(nn.Module):
    def __init__(self, model_name, classes_num):
        super().__init__()
        
        # Model Encoder
        self.net = timm.create_model(model_name, pretrained=True, in_chans=3)
        self.avg_pool = GeM()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.net.classifier.in_features, classes_num, bias=True)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.fc1)
        
    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout1(x)
        x = self.fc1(x)
        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128, num_class=9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(5)
        # self.dropout = nn.Dropout2d()
        # self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.avgpool(x)
        # x = self.dropout(x)
        # x = x.view(-1, 128*5*5)
        # x = F.relu(self.fc(x))
        # x = self.classifier(x)

        return x


def train_loop(fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    train_df = pd.read_csv('inputs/folds.csv')
    
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold == fold]

    train_dataset = TripletDataset(train_fold, CFG.TRAIN_IMG_PATH, transform['train'])
    valid_dataset = TripletDataset(valid_fold, CFG.TRAIN_IMG_PATH, transform['val'])


    train_loader = torch.utils.data.DataLoader(
                    train_dataset, shuffle=True, 
                    batch_size=CFG.TRAIN_BATCH_SIZE,
                    num_workers=0, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, shuffle=False, 
                    batch_size=CFG.VALID_BATCH_SIZE,
                    num_workers=0, pin_memory=True)

    del train_dataset, valid_dataset; gc.collect()

    # model = EffnNet(CFG.MODEL_NAME, len(label_cols))
    model = TripletNet()
    model = model.to(CFG.DEVICE)

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # ====================================================
    # apex
    # ====================================================
    if CFG.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    patience = CFG.PATIENCE
    p = 0
    min_loss = 999
    best_score = -np.inf

    for epoch in range(1, CFG.EPOCHS + 1):

        LOGGER.info("Starting {} epoch...".format(epoch))

        start_time = time.time()
        
        train_loss = train_fn(model, train_loader, CFG.DEVICE, criterion, optimizer, scheduler)
        valid_loss = valid_fn(model, valid_loader, CFG.DEVICE, criterion)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        # LOGGER.info(f"Epoch {epoch+1} - train_AUC:{train_avg['AUC']:0.5f}  valid_AUC:{valid_avg['AUC']:0.5f}")

        # if valid_avg['AUC'] > best_score:
        if valid_loss < min_loss:
            # LOGGER.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['AUC']}")
            LOGGER.info(f">>>>>>>> Loss Improved From {min_loss} ----> {valid_loss}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            # best_score = valid_avg['AUC']
            min_loss = valid_loss
            p = 0 

        if p > 0: 
            LOGGER.info(f'val loss is not updated while {p} epochs of training')
        p += 1
        if p > patience:
            LOGGER.info(f'Early Stopping')
            break


def get_oof(fold):
    LOGGER.info(f"========== fold: {fold} gen_oof ==========")
    train_df = pd.read_csv('inputs/folds.csv')

    valid_fold = train_df[train_df.kfold == fold]
    valid_dataset = TripletDataset(valid_fold, CFG.TRAIN_IMG_PATH, transform['val'])
    valid_loader = torch.utils.data.DataLoader(
                    valid_dataset, shuffle=False,
                    batch_size=CFG.VALID_BATCH_SIZE,
                    num_workers=0, pin_memory=True)

    del valid_dataset; gc.collect()

    model = EffnNet(CFG.MODEL_NAME, len(label_cols))
    model = model.to(CFG.DEVICE)
    model.load_state_dict(torch.load(OUTPUT_DIR+f'fold-{fold}.bin'))

    new_label_cols = [f'oof_{c}' for c in label_cols]
    val_preds = inference_fn(model, valid_loader, CFG.DEVICE)
    tmp_df = pd.DataFrame(val_preds, columns=new_label_cols)
    tmp_df['StudyInstanceUID'] = X_valid_idx
    _oof_df = pd.merge(valid_fold, tmp_df, on='StudyInstanceUID')
    _cv = roc_auc_score(_oof_df[label_cols], _oof_df[new_label_cols])
    LOGGER.info(f"fold {fold} AUC : {_cv}")
    return _oof_df


def calc_overall_cv():
    oof = []
    for use_fold in range(5):
        _oof = get_oof(use_fold)
        oof.append(_oof)
    oof = pd.concat(oof, 0)
    cv = roc_auc_score(oof[label_cols], oof[[f'oof_{c}' for c in label_cols]])
    LOGGER.info(f'overall AUC : {cv}')
    oof.to_csv(OUTPUT_DIR+"oof.csv", index=False)


def create_folds():
    train = pd.read_csv(CFG.TRAIN_PATH)
    train['lower_title'] = train['title'].map(lambda x: x.lower())

    groups = train['label_group'].values
    kf = GroupKFold(n_splits=5)

    train['kfold'] = -1
    for fold_id, (trn_idx, val_idx) in enumerate(kf.split(train,None,groups)):
        train.loc[val_idx, 'kfold'] = fold_id
    
    train['kfold'] = train['kfold'].astype(int)
    train.to_csv('inputs/folds.csv', index=False)


# create_folds()

for use_fold in range(5):
    train_loop(use_fold)

# calc_overall_cv()

