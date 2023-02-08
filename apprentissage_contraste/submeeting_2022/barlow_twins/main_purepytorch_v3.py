import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import torchvision
import config_param
import datasets_submeeting_v2
import copy
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Precision, Recall, F1Score, Accuracy
import matplotlib.pyplot as plt

#************************************Parameters*****************************
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
SANITY_CHECK_US = config_param.Parameters.SANITY_CHECK_US
SANITY_CHECK_SU = config_param.Parameters.SANITY_CHECK_SU
PLOT_SU = config_param.Parameters.PLOTTING_SU
PLOT_US = config_param.Parameters.PLOTTING_US
print(f'checkpoint_path:{CHECKPOINT_PATH}')

projector_res18 = config_param.Parameters.Projector_res18
projector_res50 = config_param.Parameters.Projector_res50

#NUM_WORKERS = os.cpu_count()//2
NUM_WORKERS = 8
print(f'NUM_WORKERS:{NUM_WORKERS}')
print(f'checkpoint_path:{CHECKPOINT_PATH}')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'device:{device}')

cwd = Path.cwd()
path2weights = cwd / "saved_model_purtorch"
path2weights.mkdir(parents=True, exist_ok=True)
path2weights_us = path2weights/f"barlow_weights_bs{BS_US_train}_epochs100.pt"
path2weights_su = path2weights/"log_reg_weights.pt"
#********************************Tensorboard***************************************************
path_logs_us = "./runs_us/exp"
path_logs_su = "./runs_su/exp"
if TRAIN_US:
    path_log = path_logs_us
elif TRAIN_SU:
    path_log = path_logs_su
if INFERENCE == True:
    path_log = None

print(f'tensorboard writer in:{path_log}')
writer  = SummaryWriter(path_log)


# ********************** unsupervised training, with barlow model *****************************
# *********************************************************************************************
# get the unlabelled data
unlabeled_data_train = datasets_submeeting_v2.unlabeled_data_train
unlabeled_data_val  = datasets_submeeting_v2.unlabeled_data_val
# make dataloaders for the unsupervised case
train_loader_us = data.DataLoader(unlabeled_data_train, batch_size=BS_US_train, shuffle=True, num_workers=NUM_WORKERS)
val_loader_us = data.DataLoader(unlabeled_data_val, batch_size=BS_US_val, shuffle=False, num_workers=NUM_WORKERS)

print(f'len train loader:{len(train_loader_us)}, len val loader:{len(val_loader_us)}')

def testing_data():
    for x,y in train_loader_us:
        x1 = x
        x2 = y
        print('x')
        print(x.size())
        print('y')
        print(y.size())
        break
# helper function
def off_diagonal(x):
    """ return a flattened view of the off-diagonal elements of a square matrix
    need for the barlow model implementation """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, lambd=LAMBD, weight_decay=WEIGHT_DECAY_US, arch=ARCH):
        super().__init__()
        #self.args = args
        self.lambd = lambd
        self.weight_decay = weight_decay

        if arch=="res18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)
            projector = projector_res18
        elif arch=="res34":
            self.backbone = torchvision.models.resnet34(zero_init_residual=True)
            projector = projector_res18
        elif arch=="res50":
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            projector = projector_res50

        if arch=="res18" or arch=="res34":
            sizes = [512] + list(map(int, projector.split('-')))     #for resnet18 or resnet34
        elif arch=="res50":
            sizes = [2048] + list(map(int, projector.split('-')))      #for resnet50

        self.backbone.fc = nn.Identity()

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        # empirical cross correrlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss

# test of the model

def testmodel(x1,x2):
    M1 = BarlowTwins()
    print(M1)
    out = M1(x1,x2)
    print(out)
Model = BarlowTwins()

# define optimizer
def check_device(model):
    return next(model.parameters()).is_cuda

# compute the loss per batch
# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_batch(loss_func, y1, y2, opt=None):
    """compute the loss per batch"""
    loss = loss_func(y1, y2)
    # if metric
    # with torch.no_grad():
    #    metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()

def loss_epoch(loss_func, dataset_dl, sanity_check=False, opt=None):
    """ compute the loss  over the entire dataset, because of barlow model,
     the output of the model give directly the loss of 2 augmented images, ie loss_func=model !"""
    running_loss = 0.0
    len_data = len(dataset_dl.dataset)

    for step, (y1,y2) in enumerate(dataset_dl):
    #for y1, y2 in dataset_dl:
        #print(step)
        y1 = y1.to(device)
        y2 = y2.to(device)
        loss = loss_batch(loss_func, y1, y2)
        running_loss += loss
        if sanity_check:
            break
    res = running_loss / float(len_data)
    print(f"step:{step}, running_loss:{res}")
    return res

def train_val(params):
    """ main function for training and evaluating the model"""

    num_epochs  = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    valid_dl = params["valid_dl"]
    sanity_check_us = params["sanity_check_us"]
    path2weights_us = params["path2weights_us"]
    loss_history = {"train": [], "val": []}
    # make a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    # initalize the best loss to infinite
    best_loss = float("inf")

    # define the loop to calculate the loss over one epoch
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f'Epoch {epoch}/{num_epochs-1}, current lr={current_lr}')
        model.train()
        train_loss = loss_epoch(loss_func, train_dl, sanity_check_us,opt)
        loss_history["train"].append(train_loss)
        writer.add_scalar("Loss_train", train_loss, epoch)
        # evaluate on the validation dataset
        model.eval()
        with torch.no_grad():
            valid_loss = loss_epoch(loss_func, valid_dl, sanity_check_us, opt=None)
            loss_history["val"].append(valid_loss)
            writer.add_scalar("Loss_valid", valid_loss, epoch)
        # store best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # store weights in a local file
            torch.save(model.state_dict(), path2weights_us)
            print("copied best model weights")
        lr_scheduler.step(valid_loss)
        if lr_scheduler !=get_lr(opt):
            print("loading best model weights!")
            model.load_state_dict(best_model_wts)
    model.load_state_dict(best_model_wts)

    return model, loss_history

model = Model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_US, weight_decay=WEIGHT_DECAY_US)
lr_scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=10, verbose=1)
params_us = {"num_epochs":EPOCHS_US,
          "loss_func":model,
          "optimizer":optimizer,
          "train_dl":train_loader_us,
          "valid_dl":val_loader_us,
          "sanity_check_us":SANITY_CHECK_US,
          "path2weights_us":path2weights_us}

if TRAIN_US:
    model, loss_history = train_val(params_us)

# Train - validation progess

# plot loss progress
def plotting(loss_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Loss")
    plt.plot(range(1, epochs + 1), loss_history["train"], label="train")
    plt.plot(range(1, epochs + 1), loss_history["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_metric(metric_history, epochs):
    plt.subplot(411)
    plt.title("Train-val Metric")
    plt.plot(range(1, epochs + 1), metric_history["prec_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["prec_val"], label="val")
    plt.ylabel("Prec")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.subplot(412)
    #plt.title("Train-val Metric")
    plt.plot(range(1, epochs + 1), metric_history["rec_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["rec_val"], label="val")
    plt.ylabel("Rec")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.subplot(413)
    #plt.legend()
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_val"], label="val")
    plt.ylabel("Accu_multi")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.subplot(414)
    plt.plot(range(1, epochs + 1), metric_history["accu_samples_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["accu_samples_val"], label="val")
    plt.ylabel("Accu_multi")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_prec(metric_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Prec")
    plt.plot(range(1, epochs + 1), metric_history["prec_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["prec_val"], label="val")
    plt.ylabel("Prec")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_rec(metric_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Rec")
    plt.plot(range(1, epochs + 1), metric_history["rec_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["rec_val"], label="val")
    plt.ylabel("Rec")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_acc_multi(metric_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Accu multi")
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_val"], label="val")
    plt.ylabel("Rec")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_acc_multi(metric_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Accu multi")
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["accu_multi_val"], label="val")
    plt.ylabel("Accu_multi")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

def plotting_acc_samples(metric_history, epochs):
    plt.subplot(211)
    plt.title("Train-Val Accu samples")
    plt.plot(range(1, epochs + 1), metric_history["accu_samples_train"], label="train")
    plt.plot(range(1, epochs + 1), metric_history["accu_samples_val"], label="val")
    plt.ylabel("Accu_samples")
    plt.xlabel("Training epochs")
    plt.legend()
    plt.show()

if PLOT_US:
    plotting(loss_history, EPOCHS_US)

#******************************** Prepare the features****************************
#*********************************************************************************

@torch.no_grad()
def prepare_data_feature(model, dataset, bs, shuffle=None):
    # prepare model
    network = deepcopy(model.backbone)
    network.eval()
    network.to(device)
    print(f"len dataset:{len(dataset)}")
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
    #print('***done with batch***')
    print("feats")
    print(feats.size())
    print("labels")
    print(labels.size())

    return data.TensorDataset(feats, labels)

# loading the model
def loading_model(arch=ARCH, device=device):
    """ load the best self supervised model obtained """
    #path_model_res18 = Path('/media/ben/Data_linux/code/submeeting_code/code/saved_model_purtorch/')
    #path_model_res18 = Path('/home/lepersbe/Narval/submeeting_2022/code/saved_model_purtorch/')
    path_model_res18 = Path('/mnt/narval/narval_BL/contrastive_learning/Narval/saved_model_purtorch/')
    #str_res18 = "barlow_weights.pt"
    str_res18 =  "barlow_weights_bs256_epochs100.pt"

    if arch=="res18":
        path_model_best = path_model_res18/str_res18
    elif arch=="res50":
        path_model_best = path_model_res50/str_res50

    model_contrast = BarlowTwins()
    model_contrast.load_state_dict(torch.load(path_model_best, map_location=device))
    model_contrast.eval()

    return model_contrast

# define a logistic model for the supervised part
### LogisticRegressionModel
class LogisticRegression(nn.Module):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=EPOCHS_SU, thresh=0.5):
        super().__init__()
        #self.save_hyperparameters()
        # mapping from representation h to class
        self.model = nn.Linear(feature_dim, num_classes)
        self.thresh = thresh

    def forward(self, x):
        #print(f"x size\n:{x.size()}")
        return self.model(x)

def testing_log_model():
    xtest = torch.rand((8,512))
    L = LogisticRegression(512, 7, LR_US, WEIGHT_DECAY_US)
    outest = L(xtest)
    print("outest")
    print(outest.size())

#testing_log_model()
# get learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target, thres=0.5):
    """ compute metrics """
    #pred = output.argmax(dim=1, keepdim=True)
    #preds = (((output.sigmoid() > thres).long()).cpu()).unsqueeze(1)
    preds = (((output.sigmoid() > thres).long()).cpu())
    lbls = ((target.long()).cpu())

    precision = Precision(average='macro', num_classes=NUM_CLASSES, mdmc_average='global', multiclass=True, task="binary")
    recall = Recall(average='macro', num_classes=NUM_CLASSES, mdmc_average='global', multiclass=True, task="binary")
    F1 = F1Score(average='macro', num_classes=NUM_CLASSES, mdmc_average='global', multiclass=True, task='binary')

    Accu_scikit = Accuracy(subset_accuracy=True,task="binary")


    acc_multi = (preds == lbls).float().mean()   # equivalent to  Accu_scikit2 = Accuracy(subset_accuracy=False)

    prec = precision(preds, lbls)
    rec = recall(preds, lbls)
    f1 = F1(preds, lbls)
    accu_scikit = Accu_scikit(preds, lbls)

    return (prec.item(), rec.item(), f1.item(), acc_multi.item(), accu_scikit.item())

def loss_batch_log(loss_func,output, target, opt=None):
    """ compute the loss per batch """
    loss = loss_func(output, target)

    with torch.no_grad():
        metrics_tuple = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metrics_tuple

def loss_epoch_log(model, loss_func, dataset_dl, sanity_check=SANITY_CHECK_SU, opt=None,writer=writer):
    """ compute the loss for the entire dataset"""
    running_loss = 0.0
    running_prec = 0.0
    running_rec  = 0.0
    running_f1 = 0.0
    running_acc_multi = 0.0
    running_accu_samples= 0.0

    len_data = len(dataset_dl)

    for niter, (xb, yb) in enumerate(dataset_dl):
        # move batch to device
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        # get loss per barch
        loss_b, metrics_tuple = loss_batch_log(loss_func, output, yb, opt)
        # update running loss
        running_loss += loss_b
        # if metrics_tuple is not None:
        running_prec = running_prec + metrics_tuple[0]
        running_rec  = running_rec + metrics_tuple[1]
        running_f1   = running_f1 + metrics_tuple[2]
        running_acc_multi = running_acc_multi + metrics_tuple[3]
        running_accu_samples = running_accu_samples + metrics_tuple[4]

        if model.training :
            writer.add_scalar('Loss train', loss_b, niter)
            writer.add_scalar('Prec train', metrics_tuple[0], niter)
            writer.add_scalar('Rec train', metrics_tuple[1], niter)
            writer.add_scalar('F1 train', metrics_tuple[2], niter)
            writer.add_scalar('acc multi train', metrics_tuple[3], niter)
            writer.add_scalar('acc samples train', metrics_tuple[4], niter)
        else:
            writer.add_scalar('Loss val', loss_b, niter)
            writer.add_scalar('Prec val', metrics_tuple[0], niter)
            writer.add_scalar('Rec val', metrics_tuple[1], niter)
            writer.add_scalar('F1 val', metrics_tuple[2], niter)
            writer.add_scalar('acc multi val', metrics_tuple[3], niter)
            writer.add_scalar('acc samples val', metrics_tuple[4], niter)

        if sanity_check:
            break
    # average the loss value
    loss = running_loss / float(len_data)
    # average metrics value
    prec = running_prec / float(len_data)
    rec = running_rec / float(len_data)
    f1 = running_f1 / float(len_data)
    accu_multi = running_acc_multi / float(len_data)
    accu_samples = running_accu_samples / float(len_data)


    return loss, (prec, rec, f1, accu_multi, accu_samples)

#def train_val_log(model, params_su,weight_decay=WEIGHT_DECAY_SU,lr=LR_SU):
def train_val_log(params_su):

    """ main function for training and evaluating the model on the dataset"""

    num_epochs = params_su["number_epochs"]
    loss_func = params_su["loss_func"]
    opt_su = params_su["optimizer"]
    train_dl = params_su["train_dl"]
    val_dl = params_su["val_dl"]
    sanity_check = params_su["sanity_check"]
    path2weights_su = params_su["path2weights_su"]
    model = params_su["model_su"]

    # history of loss values
    loss_history = {"train": [], "val": []}
    # history of metrics
    metrics_history = {"prec_train":[],
                       "prec_val" : [],
                       "rec_train":[],
                       "rec_val": [],
                       "f1_train":[],
                       "f1_val":[],
                       "accu_multi_train":[],
                       "accu_multi_val":[],
                       "accu_samples_train":[],
                       "accu_samples_val":[]
                       }

    # make a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())

    # initialize the best loss to a large value
    best_loss = float("inf")
    # loop to calculate the loss over 1 epoch
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = get_lr(opt_su)
        print(f'Epoch {epoch}/{num_epochs-1}, current lr={current_lr}')
        # train model on the current training dataset
        model.train()
        train_loss, train_metrics = loss_epoch_log(model, loss_func, train_dl, SANITY_CHECK_SU, opt_su)
        print(f"train loss:{train_loss}")
        # collect loss and metrics for the training dataset
        loss_history["train"].append(train_loss)
        metrics_history["prec_train"].append(train_metrics[0])
        metrics_history["rec_train"].append(train_metrics[1])
        metrics_history["f1_train"].append(train_metrics[2])
        metrics_history["accu_multi_train"].append(train_metrics[3])
        metrics_history["accu_samples_train"].append(train_metrics[4])

        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = loss_epoch_log(model, loss_func, val_dl, SANITY_CHECK_SU)
            # collect loss and metrics for the training dataset
            loss_history["val"].append(val_loss)
            metrics_history["prec_val"].append(val_metrics[0])
            metrics_history["rec_val"].append(val_metrics[1])
            metrics_history["f1_val"].append(val_metrics[2])
            metrics_history["accu_multi_val"].append(val_metrics[3])
            metrics_history["accu_samples_val"].append(val_metrics[4])

            # store best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts  = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights_su)
                print("copied best model weigths")


    model.load_state_dict(best_model_wts)

    return model, loss_history, metrics_history


if TRAIN_SU:
    print(f"device:{device}")
    model_contrast = loading_model(arch="res18", device=device)
    print("loading the self supervised model to build the features")
    #print(model_contrast)

    #opt = torch.optim.Adam(model_contrast.parameters(), lr=LR_SU, weight_decay=WEIGHT_DECAY_SU)
    # gets the train and val dataset for supervised learning
    train_ds = datasets_submeeting_v2.train_ds
    val_ds = datasets_submeeting_v2.val_ds

    train_feats_barlow = prepare_data_feature(model_contrast, train_ds, bs=BS_SU_train, shuffle=True)
    val_feats_barlow = prepare_data_feature(model_contrast, val_ds, bs=BS_SU_val, shuffle=False)

    train_feats_barlow_dl = data.DataLoader(train_feats_barlow, batch_size=BS_SU_train, shuffle=True, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    val_feats_barlow_dl = data.DataLoader(val_feats_barlow, batch_size=BS_SU_val, shuffle=False, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    print(f'len train_feats_barlow:{len(train_feats_barlow)}')
    print(f'len val_feats_barlow:{len(val_feats_barlow)}')
    print(type(train_feats_barlow))
    print(f'feats_train X shape:{train_feats_barlow[0][0].shape}, feats y shape:{train_feats_barlow[0][1].shape}')
    print(f"train_feats_barlow_dl:{len(train_feats_barlow_dl)}")
    for x in next(iter(train_feats_barlow_dl)):
        print("x")
        print(x.size())

    feature_dim = train_feats_barlow.tensors[0].shape[1],
    print(f"feature_dim:{feature_dim}")
    print(feature_dim[0])
    model_reg = LogisticRegression(feature_dim=feature_dim[0],
                                   num_classes=NUM_CLASSES,
                                   lr=LR_SU,
                                   weight_decay= WEIGHT_DECAY_SU,
                                   max_epochs=EPOCHS_SU)

    print(model_reg)
    optimizer_su = torch.optim.Adam(model_reg.parameters(), lr=LR_SU, weight_decay=WEIGHT_DECAY_SU)

    params_su = {"number_epochs": EPOCHS_SU,
              "loss_func": torch.nn.BCEWithLogitsLoss(),
              "train_dl": train_feats_barlow_dl,
              "val_dl": val_feats_barlow_dl,
              "sanity_check": SANITY_CHECK_SU,
              "path2weights_su": path2weights_su,
              "model_su":model_reg.to(device),
              "optimizer":optimizer_su}

    model_regr, loss_history, metrics_history = train_val_log(params_su)
    print(f"loss_history train:{len(loss_history['train'])}, loss_history_val:{len(loss_history['val'])}")
    print(f"train loss:\n{loss_history['train']}")
    print(f"val loss:\n{loss_history['val']}")
    print("metrics")
    print(metrics_history)
    if PLOT_SU:
        plotting(loss_history, EPOCHS_SU)
        #plotting_metric(metrics_history, EPOCHS_SU)
        plotting_prec(metrics_history, EPOCHS_SU)
        plotting_rec(metrics_history, EPOCHS_SU)
        plotting_acc_multi(metrics_history, EPOCHS_SU)
        plotting_acc_samples(metrics_history, EPOCHS_SU)
