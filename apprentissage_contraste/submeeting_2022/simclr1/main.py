import os
from pathlib import Path
import matplotlib
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
from sklearn.model_selection import train_test_split
import make_datasets
import config_param
import torchmetrics
from torchmetrics.utilities import check_forward_full_state_property

from torchmetrics.classification import MultilabelExactMatch

# control boolean variable
TRAIN_US = config_param.Parameters.TRAIN_US
TRAIN_SU = config_param.Parameters.TRAIN_SU
INFERENCE = config_param.Parameters.INFERENCE
# parameters
# number of augmentations per images
N_VIEWS = config_param.Parameters.N_VIEWS
# Batch_size for unsupervised and supervised training
BS_SU_train = config_param.Parameters.BS_SU_train
BS_SU_val = config_param.Parameters.BS_SU_val
BS_US_train = config_param.Parameters.BS_US_train
BS_US_val = config_param.Parameters.BS_US_val
EPOCHS_US = config_param.Parameters.EPOCHS_US
EPOCHS_SU = config_param.Parameters.EPOCHS_SU

LR_US = config_param.Parameters.LR_US
LR_SU = config_param.Parameters.LR_SU
TEMPERATURE = config_param.Parameters.TEMPERATURE
WEIGHT_DECAY_US = config_param.Parameters.WEIGHT_DECAY_US
WEIGHT_DECAY_SU = config_param.Parameters.WEIGHT_DECAY_SU
HIDDEN_DIM = config_param.Parameters.HIDDEN_DIM
NUM_CLASSES = config_param.Parameters.NUM_CLASSES
GAMMA = config_param.Parameters.GAMMA
#NUM_WORKERS = os.cpu_count()//2
NUM_WORKERS=8
print(f'NUM_WORKERS:{NUM_WORKERS}')
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ContrastiveLearning/")
print(f'checkpoint_path:{CHECKPOINT_PATH}')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'device:{device}')

# get the unlabelled data
unlabeled_data_train = make_datasets.unlabeled_data_train
unlabeled_data_val  = make_datasets.unlabeled_data_val

def check1():
    for i in range(N_VIEWS):
        print(unlabeled_data_train[0][i].shape)

#check1()

def plot_test():

    im1, im2 = next(iter(unlabeled_data_train))
    print(im1.shape)
    print('***')
    print(im2.shape)
    imgs_test = torch.stack([im1, im2],dim=0)
    imggrid = torchvision.utils.make_grid(imgs_test, nrow=2, normalize=True, pad_value=0.9)
    imggrid = imggrid.permute(1,2,0)
    plt.imshow(imggrid)
    plt.show()

#plot_test()
# make dataloaders for the unsupervised case
train_loader_us = data.DataLoader(unlabeled_data_train, batch_size=BS_US_train, shuffle=True, num_workers=NUM_WORKERS)
val_loader_us = data.DataLoader(unlabeled_data_val, batch_size=BS_SU_val, shuffle=False, num_workers=NUM_WORKERS)

def check():
    print(f'len unlabeled_loader_train:{len(train_loader_us)}')
    print(f'len unlabeled_loader_train:{len(val_loader_us)}')
    batch_0 = next(iter(train_loader_us))
    print(f'len(batch_0):{len(batch_0)}')
    for item in batch_0:
        print(item.shape)
        im1 = item
    imgs = batch_0

    imgs_cat = torch.cat((imgs), dim=0)
    print(f'imags_cat:{imgs_cat.shape}')
    return imgs_cat, im1

imgs_inpu, im1 = check()

def accuracy_multi_modif(inp, targ, thresh=0.5, sigmoid=False):
    "Compute accuracy when `inp` and `targ` are the same size."
    #inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()

    return (((targ == inp).all(dim=1)).sum())/targ.shape[0]


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=EPOCHS_US, nviews=N_VIEWS):
        super().__init__()
        self.save_hyperparameters()
        self.nviews = nviews
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(pretrained=False, num_classes=4 * hidden_dim)
        # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                                        nn.ReLU(inplace=True),
                                        nn.Linear(4 * hidden_dim, hidden_dim),)   # (512, 128)

    def forward(self, x):
        return self.convnet(x)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        img = batch
        imgs = torch.cat(img, dim=0)


        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // self.nviews, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


# check input -  output dimensions of the model
sim_clr_model = SimCLR(hidden_dim=HIDDEN_DIM,
                       lr=LR_US,
                       temperature=TEMPERATURE,
                       weight_decay=WEIGHT_DECAY_US,
                       nviews=N_VIEWS)
print("******Contrastive Model******")
#print(sim_clr_model)
print(f'input:{im1.size()}')
output = sim_clr_model(im1)
print(f'output:{output.size()}')


# training step for the contrastive model
def train_simclr(max_epochs, hidden_dim, lr,temperature, weight_decay, train_loader_us, val_loader_us):

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
                                    LearningRateMonitor("epoch"),],progress_bar_refresh_rate=1,)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")

    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:

        pl.seed_everything(42)  # To be reproducable
        model = SimCLR(hidden_dim, lr, temperature, weight_decay, max_epochs)
        trainer.fit(model, train_loader_us, val_loader_us)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

# load the contrastive model from previous runs
def loading_model():

    path_model = Path('/mnt/narval/narval_BL/submeeting_2022/simclr1/saved_models/ContrastiveLearning/SimCLR/lightning_logs/version_2/')
    model_best = 'epoch=774-step=23249.ckpt'
    path_model_best = path_model/model_best
    simclr_model = SimCLR.load_from_checkpoint(path_model_best)

    return simclr_model

if TRAIN_US:
    simclr_model = train_simclr(max_epochs=EPOCHS_US,
                                hidden_dim=HIDDEN_DIM,
                                lr=LR_US,
                                temperature=TEMPERATURE,
                                weight_decay=WEIGHT_DECAY_US,
                                train_loader_us=train_loader_us,
                                val_loader_us=val_loader_us
                                )
else:
    simclr_model = loading_model()

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
                                weight_decay=self.hparams.weight_decay)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.6),
                                                                  int(self.hparams.max_epochs * 0.8)],
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
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()
    network.eval()
    network.to(device)

    # encode all the images
    data_loader = data.DataLoader(dataset, batch_size=bs, num_workers=NUM_WORKERS, shuffle=shuffle, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    return data.TensorDataset(feats, labels)

model_logistic = LogisticRegression(feature_dim=128, num_classes=NUM_CLASSES, lr=LR_SU, weight_decay=WEIGHT_DECAY_SU)

#simclr = loading_model()

train_ds = make_datasets.train_ds
val_ds = make_datasets.val_ds

def make_features():
    train_feats_simclr = prepare_data_feature(simclr_model, train_ds, bs=BS_SU_train, shuffle=True)
    val_feats_simclr = prepare_data_feature(simclr_model, val_ds, bs=BS_SU_val, shuffle=False)

    print(f'train_feats_simclr:{len(train_feats_simclr)}')
    print(f'val_feats_simclr:{len(val_feats_simclr)}')
    print(f'feats_train X shape:{train_feats_simclr[0][0].shape}, feats y shape:{train_feats_simclr[0][1].shape}')

    return train_feats_simclr, val_feats_simclr

if TRAIN_SU:
    train_feats_simclr, val_feats_simclr = make_features()
#train_loop for the supervised model


def train_logreg(bs_train, bs_val, train_feats_data, val_feats_data, model_suffix, max_epochs=EPOCHS_SU, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch"),],)

    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=bs_train, shuffle=True, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(val_feats_data, batch_size=bs_val, shuffle=False, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

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

def load_logistic():

    path_model = Path('/mnt/narval/narval_BL/submeeting_2022/simclr1/saved_models/ContrastiveLearning/LogisticRegression/lightning_logs/version_2/checkpoints')
    #model_best = 'epoch=0-step=72.ckpt'
    model_best = 'epoch=3-step=291.ckpt'
    path_model_best = path_model/model_best
    logistic_model = LogisticRegression.load_from_checkpoint(path_model_best)
    return logistic_model


def supervised_training(lr=LR_SU, wd=WEIGHT_DECAY_SU):
    logistic_model, results = train_logreg(
        bs_train=BS_SU_train,
        bs_val=BS_SU_val,
        train_feats_data=train_feats_simclr,
        val_feats_data=val_feats_simclr,
        model_suffix=100,
        feature_dim=train_feats_simclr.tensors[0].shape[1],
        num_classes=NUM_CLASSES,
        lr=lr,
        weight_decay=wd,)
    return logistic_model, results

if TRAIN_SU:
    #for lr in LR_SU:
    logistic_model,results = supervised_training()
else:
    logistic_model = load_logistic()


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
        pred = (pred_features.sigmoid() > 0.5).float()
        print('prediction')
        print(pred.size())
        print(pred)
        print('target')
        print(y.size())
        print(y)
#inference(val_loader = test_loader)

def inference_image_to_class(nb=6):
    my_net = deepcopy(simclr_model.convnet)
    my_net.fc = nn.Identity()          # remove projection head  , it means we remove f, so the output dimensions ar 4*hidden_dim
    my_net.eval()
    my_net

    data_loader = data.DataLoader(val_ds, batch_size=BS_SU_val, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)

    count=0
    for batch_imgs, batch_labels in tqdm(data_loader):
        count+=1

        batch_feats = my_net(batch_imgs)
        batch_classes = ((logistic_model(batch_feats)).sigmoid() > 0.5).float()
        print('predicted_classes:',end='\n')
        print(batch_classes)
        print('***************')
        print('labels',end='\n')
        print(batch_labels)
        imggrid = torchvision.utils.make_grid(batch_imgs, nrow=4, normalize=True, pad_value=0.9)
        imggrid = imggrid.permute(1,2,0)
        plt.imshow(imggrid)
        plt.show()
        if count==nb:
            break
if INFERENCE:
    inference_image_to_class()

