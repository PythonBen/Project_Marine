import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# metrics
def accuracy_multi(labels, preds, thresh=0.5):
    acc_multi = (((preds.sigmoid() > thresh) == labels.bool()).float().mean())
    acc_multi.detach().cpu()
    return acc_multi.item()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def precision(labels, preds, thresh=0.5):
    """ precision: (# items correctry predicted)/(# items attributed to class i) """

    if len(preds) !=0:
        preds = preds.detach().cpu().numpy()
        predictions = np.array((sigmoid(preds) > thresh), dtype=int)
        labels = labels.detach().cpu().numpy()
        return precision_score(labels, predictions, average="macro", zero_division=0)
    else:
        raise ValueError('Error in predictions')

def recall(labels, preds, thresh=0.5):
    """ recall: (# items correctly predicted)/(# items belongings to class i)"""

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    predictions = np.array((sigmoid(preds) > thresh), dtype=int)
    return recall_score(labels, predictions, average="macro", zero_division=0)

def F1_score(labels, preds, thresh=0.5):

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    predictions = np.array((sigmoid(preds) > thresh), dtype=int)
    return f1_score(labels, predictions, average="macro", zero_division=0)

def train(dataloader, model, optimizer, device):

    """
    This function does the training for one epoch
    :param dataloder: pytorch dataloder
    :param model: pytorch model
    :param optimizer:  optimizer
    :param device: cuda or cpu
    :return:
    """
    loss_batch = []
    # put the model in train mode
    model.train()
    # go over every batch from the dataloader
    for step, data in enumerate(dataloader):
        # we have image and targets in our dataset
        inputs = data["image"]
        targets = data["targets"]
        # move inputs and target to the device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # zero grad the optimizer
        optimizer.zero_grad()
        # forward step
        outputs = model(inputs)
        # calculate the loss, here it is multi label classification , we need BCWewith logits loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        # backpropagation
        loss.backward()
        # step the optimizer
        optimizer.step()
        # rate scheduler here if necessary
        loss_batch.append(loss.item())

    return sum(loss_batch)/len(loss_batch)

def evaluate(data_loader, model, device):
    """
    This function does evaluation for one epoch
    :param data_loader:
    :param model:
    :param device:
    :return:
    """
    model.eval()
    # lists to store targets and outputs
    # we use no grad context manager
    loss_batch = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    F1_list  = []

    with torch.no_grad():
        for data in data_loader:
            ldat = len(data_loader)
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # forward loop to generate predictions
            output = model(inputs)
            # compute the loss
            loss = nn.BCEWithLogitsLoss()(output, targets)
            # compute the metrics
            acc  = accuracy_multi(targets, output)
            prec = precision(targets, output)
            rec = recall(targets, output)
            f1 = F1_score(targets, output)
            # make lists
            loss_batch.append(loss.item())
            accuracy_list.append(acc)
            precision_list.append(prec)
            recall_list.append(rec)
            F1_list.append(f1)

    return sum(loss_batch)/ldat, sum(accuracy_list)/ldat, sum(precision_list)/ldat, \
           sum(recall_list)/ldat, sum(F1_list)/ldat
