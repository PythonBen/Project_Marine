""""
@author: lepersbe
Implementation of the Barlow twin model with pytorch lightning
Ressources:
https://pytorchlightning.github.io/lightning-tutorials/notebooks/course_UvA-DL/13-contrastive-learning.html
paper: Barlow Twins: Self-Supervised Learning via Redundancy Reduction (J. Zbontar, L.Jing,I.Misra, Y.LeCun, S.Deny)
 (https://arxiv.org/abs/2103.03230)
code: https://github.com/facebookresearch/barlowtwins
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

import config_param
import datasets_submeeting_v2
from torchmetrics import Precision, Recall, F1Score, Accuracy

from torchmetrics.classification import MultilabelExactMatch, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score



# control boolean variable to train the unsupervised model backbone or the supervised model head
TRAIN_US = config_param.Parameters.TRAIN_US
TRAIN_SU = config_param.Parameters.TRAIN_SU
INFERENCE = config_param.Parameters.INFERENCE
ARCH = config_param.Parameters.ARCH
# parameters
N_VIEWS = config_param.Parameters.N_VIEWS
BS_US_train = config_param.Parameters.BS_US_train
BS_US_val = config_param.Parameters.BS_US_val
BS_SU_train = config_param.Parameters.BS_SU_train
BS_SU_val = config_param.Parameters.BS_SU_val
EPOCHS_US = config_param.Parameters.EPOCHS_US
EPOCHS_SU = config_param.Parameters.EPOCHS_SU
LAMBD = config_param.Parameters.LAMBD

LR_US = config_param.Parameters.LR_US
LR_SU = config_param.Parameters.LR_SU
WEIGHT_DECAY_US = config_param.Parameters.WEIGHT_DECAY_US
WEIGHT_DECAY_SU = config_param.Parameters.WEIGHT_DECAY_SU
GAMMA = config_param.Parameters.GAMMA
NUM_CLASSES = config_param.Parameters.NUM_CLASSES
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

print(f'checkpoint_path:{CHECKPOINT_PATH}')

projector_res18 = config_param.Parameters.Projector_res18
projector_res50 = config_param.Parameters.Projector_res50

#NUM_WORKERS = os.cpu_count()//2
NUM_WORKERS = 8
print(f'NUM_WORKERS:{NUM_WORKERS}')
print(f'checkpoint_path:{CHECKPOINT_PATH}')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'device:{device}')

# get the unlabelled data
unlabeled_data_train = datasets_submeeting_v2.unlabeled_data_train
unlabeled_data_val  = datasets_submeeting_v2.unlabeled_data_val
# to do: use pl to train the barlow_twin model.
def check1():
    for i in range(N_VIEWS):
        print(unlabeled_data_train[0][i].shape)
check1()
# make dataloaders for the unsupervised case
train_loader_us = data.DataLoader(unlabeled_data_train, batch_size=BS_US_train, shuffle=True, num_workers=NUM_WORKERS)
val_loader_us = data.DataLoader(unlabeled_data_val, batch_size=BS_US_val, shuffle=False, num_workers=NUM_WORKERS)

print(f'len train loader:{len(train_loader_us)}, len val loader:{len(val_loader_us)}')

for x,y in train_loader_us:
    print('x')
    print(x.size())
    print('y')
    print(y.size())
    break
def check():
    print(f'len unlabeled_loader_train:{len(train_loader_us)}')
    print(f'len unlabeled_loader_train:{len(val_loader_us)}')
    batch_0 = next(iter(train_loader_us))
    print(f'len(batch_0):{len(batch_0)}')
    list_it = []
    for item in batch_0:
        print(item.shape)
        im1 = item
        list_it.append(item)
    imgs = batch_0

    imgs_cat = torch.cat((imgs), dim=0)
    print(f'imags_cat:{imgs_cat.shape}')
    return imgs_cat, im1, list_it

#imgs_inpu, im1, list_it = check()

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Barlow(pl.LightningModule):
    def __init__(self,lr, weight_decay, max_epochs=EPOCHS_US, lambd=LAMBD, arch=ARCH):
        super().__init__()
        self.save_hyperparameters()
        if arch=="res18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)
            projector = projector_res18
        elif arch=="res34":
            self.backbone = torchvision.models.resnet34(zero_init_residual=True)
            projector = projector_res18
        elif arch=="res50":
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            projector = projector_res50
        self.backbone.fc = nn.Identity()
        self.lambd = lambd
        self.weight_decay = weight_decay

        if arch=="res18" or arch=="res34":
            sizes = [512] + list(map(int, projector.split('-')))     #for resnet18 or resnet34
        elif arch=="res50":
            sizes = [2048] + list(map(int, projector.split('-')))      #for resnet50

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def loss_barlow(self,batch, mode="train"):
        x1, x2 = batch
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        # empirical cross correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        loss = loss.mean()
        self.log(mode + "_loss", loss)
        return loss

    def configure_optimizers(self):
        #optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.weight_decay)
        #optimizer.param_groups()
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)

        #return [optimizer], [lr_scheduler]
        return [optimizer]


    def training_step(self, batch, batch_idx):

        return self.loss_barlow(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.loss_barlow(batch, mode="val")


# training with pytorch lightning
def train_barlow(lr, weight_decay, max_epochs, train_loader_us, val_loader_us):

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "Barlow_twins"),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=max_epochs,
                         log_every_n_steps=10,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                    LearningRateMonitor("epoch"),])
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "Barlow.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = Barlow.load_from_checkpoint(pretrained_filename)
    else:

        pl.seed_everything(42)  # To be reproducable
        model = Barlow(lr=lr,weight_decay=weight_decay, max_epochs=max_epochs)
        trainer.fit(model, train_loader_us, val_loader_us)
        # Load best checkpoint after training
        model = Barlow.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

if TRAIN_US:
    barlow_model = train_barlow(lr=LR_US,
                                weight_decay=WEIGHT_DECAY_US,
                                max_epochs=EPOCHS_US,
                                train_loader_us=train_loader_us,
                                val_loader_us=val_loader_us)


# define a logistic model for the supervised part
class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=EPOCHS_SU, thresh=0.5):
        super().__init__()
        self.save_hyperparameters()
        # mapping from representation h to class
        self.model = nn.Linear(feature_dim, num_classes)
        self.thresh = thresh

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                #weight_decay = WEIGHT_DECAY_SU)
                                weight_decay=self.hparams.weight_decay)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)],
                                                      gamma=GAMMA)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch

        preds = self.model(feats)
        loss = F.binary_cross_entropy_with_logits(preds, labels)

        acc_multi = (((preds.sigmoid() > self.thresh) == labels.bool()).float().mean())

        Accu_scikit = MultilabelExactMatch(num_labels=7, threshold=0.5).to(device)
        accu_scikit = Accu_scikit(preds, labels)
        print(f"accuracy exact match:{accu_scikit:.4f}, accu_multi:{acc_multi:.4f}")

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc_multi)
        self.log(mode + "_accu_scikit", accu_scikit)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

# a small function to encode all images in our dataset
@torch.no_grad()
def prepare_data_feature(model, dataset, bs, shuffle=None):
    # prepare model
    network = deepcopy(model.backbone)
    network.eval()
    network.to(device)

    # encode all the images
    data_loader = data.DataLoader(dataset, batch_size=bs, num_workers=NUM_WORKERS, shuffle=shuffle, drop_last=False)
    feats, labels = [], []
    #print('***reading batch****')
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return data.TensorDataset(feats, labels)

def loading_model(arch=ARCH):

    path_model_res18 = Path("/mnt/narval/narval_BL/contrastive_learning/Narval/saved_models/Barlow_twins/Barlow/weights/")
    str_res18 ="epoch=7-step=1879.ckpt"

    if arch=="res18":
        path_model_best = path_model_res18/str_res18
    elif arch=="res50":
        path_model_best = path_model_res50/str_res50

    barlow_model = Barlow.load_from_checkpoint(path_model_best)

    return barlow_model

if not TRAIN_US:
    barlow_model = loading_model()

# gets the train and val dataset for supervised learning
train_ds = datasets_submeeting_v2.train_ds
val_ds = datasets_submeeting_v2.val_ds

train_feats_barlow = prepare_data_feature(barlow_model, train_ds, bs=BS_SU_train, shuffle=True)
val_feats_barlow = prepare_data_feature(barlow_model, val_ds, bs=BS_SU_val, shuffle=False)

print(f'train_feats_barlow:{len(train_feats_barlow)}')
print(f'val_feats_barlow:{len(val_feats_barlow)}')
print(f'feats_train X shape:{train_feats_barlow[0][0].shape}, feats y shape:{train_feats_barlow[0][1].shape}')

# train loop for the supervised model
def train_logreg(bs_train, bs_val, train_feats_data, val_feats_data, model_suffix, max_epochs=EPOCHS_SU, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch"),],)

    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=bs_train, shuffle=True, drop_last=False, pin_memory=True, num_workers=0)
    val_loader = data.DataLoader(val_feats_data, batch_size=bs_val, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.validate(model, dataloaders=train_loader, verbose=False)
    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    result = {"train": train_result[0]["val_acc"], "val": val_result[0]["val_acc"]}

    return model, result

def supervised_training(lr=LR_SU, wd=WEIGHT_DECAY_SU):
    logistic_model, results = train_logreg(
        bs_train=BS_SU_train,
        bs_val=BS_SU_val,
        train_feats_data=train_feats_barlow,
        val_feats_data=val_feats_barlow,
        model_suffix=100,
        feature_dim=train_feats_barlow.tensors[0].shape[1],
        num_classes=NUM_CLASSES,
        lr=lr,
        weight_decay=wd,)
    return logistic_model, results

# load the logistic regression model for inference
def load_logistic():

    path_model = Path("/mnt/narval/narval_BL/contrastive_learning/Narval/saved_models/Barlow_twins/LogisticRegression/weights/")
    saved_model = "epoch=3-step=291.ckpt"

    path_model_best = path_model/saved_model
    logistic_model = LogisticRegression.load_from_checkpoint(path_model_best)
    return logistic_model

if TRAIN_SU:
    logistic_model, results = supervised_training()
else:
    logistic_model = load_logistic()

# inference

def inference(val_loader):           # image X ->  f -> g : we use the features as the input of the logistic model
    """
    fonction to make prediction from the features to the classes
    :param val_loader: val_loader for the features
    :return: print the prediction and target classes
    """
    x, y = next(iter(val_loader))              # x: [4, 512]
    print(x.size())
    with torch.no_grad():
        pred_features = logistic_model(x)
        print(f'pred_features:{pred_features.size()}')
        #pred = ((logistic_model(pred_features)).sigmoid() > 0.5).float()
        pred = (pred_features.sigmoid() > 0.5).float()
        print('prediction')
        print(pred.size())
        print(pred)
        print('target')
        print(y.size())
        print(y)
#inference(val_loader = val_loader)

def inference_image_to_class(nb=6):
    my_net = deepcopy(barlow_model.backbone)

    my_net.eval()
    my_net

    data_loader = data.DataLoader(val_ds, batch_size=BS_SU_val, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)

    count=0
    for batch_imgs, batch_labels in tqdm(data_loader):
        count+=1
        batch_feats = my_net(batch_imgs)
        batch_classes = ((logistic_model(batch_feats)).sigmoid() > 0.5).float()
        print('predicted_classes:\n')
        print(batch_classes)
        print('***************')
        print('labels')
        print(batch_labels)
        imggrid = torchvision.utils.make_grid(batch_imgs, nrow=4, normalize=True, pad_value=0.9)
        imggrid = imggrid.permute(1,2,0)
        plt.imshow(imggrid)
        plt.show()
        if count==nb:
            break

if INFERENCE:
    inference_image_to_class()

