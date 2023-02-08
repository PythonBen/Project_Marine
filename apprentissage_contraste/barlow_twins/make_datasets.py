#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:15:28 2021

@author: lepersbe
# file to create the datasets for unsupervised training (no labels)
and supervised training (with labels)
For the labels, we use one hot encoding to encode the 3 classes (algues, rochers, sables)
"""
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config_param import Parameters

# paths to your data
path_data = Path('/home/lepersbe/Narval')
path_images = path_data/'images_data/total_images/'
path_dataframe = Path('/home/lepersbe/Narval/pickle_files')

path_im = path_images.as_posix()+'/'

# Parameters
P = Parameters()
BS_US = P.BS_US
BS_SU_train = P.BS_SU_train
BS_SU_val = P.BS_SU_val
EPOCHS_US = P.EPOCHS_US
EPOCH_SU = P.EPOCHS_SU
N_VIEWS = P.N_VIEWS
RATIO = P.RATIO
H, W  = P.H_image, P.W_image
size = (H, W)
# ****************************************************
# make a dataset with labels for Supervised learning

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)
    
    def get_x(self,i):
        '''extract the independant variable x (images)'''
        return self.img_dir + self.dataframe.iloc[i]['fname']+'.png'

    def get_y(self,i):
        '''extract the dependent variable y (labels)'''
        return torch.tensor(self.dataframe.iloc[i]['labels_encoded'])

    def __getitem__(self, idx):
        
        image = Image.open(self.get_x(idx)).convert('RGB') #PIL image
        label = self.get_y(idx) #float32

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# dataframes
frame = 'df_total.pkl'
df_raw = pd.read_pickle(path_dataframe/frame)

df_u = df_raw.drop_duplicates(subset=['fname'])

df = df_u.reset_index(drop=True)
df_copy = df.copy()

def one_hot_label_encoder(dataframe):
    str_to_int = {"algues":0,"rochers":1,"sable":2}
    Y_onehot = []
    #Y_onehot = torch.zeros((len(dataframe),3))
    for i in range(len(dataframe)):
        labels = dataframe.iloc[i]['labels'].split(' ')
        labels_coded = torch.LongTensor([str_to_int[label] for label in labels])
        
        y_onehot = nn.functional.one_hot(labels_coded, num_classes=3)
        
        y_onehot = y_onehot.sum(dim=0).float()
        y_onehot = y_onehot.numpy()
        Y_onehot.append(y_onehot)
        #Y_onehot[i] = y_onehot
    return Y_onehot

df_copy['labels_encoded']= one_hot_label_encoder(df_copy)

# do not normalize
transform_basic = transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
# dataloaders
def splitter(df):
    ''' create the train and valid indices according to the boolean "is_valid" column'''
    train = df[df['is_valid']==False]
    valid = df[df['is_valid']==True]
    return train,valid

train_df, val_df = splitter(df_copy)
train_ds = CustomImageDataset(train_df, path_im, transform=transform_basic)
val_ds = CustomImageDataset(val_df, path_im, transform=transform_basic)

# *************************
# make a unlabeled dataset for unsupersvised learning

list_images_path = []
for item in path_images.iterdir():
    list_images_path.append(path_images/item)
    
samples = len(list_images_path) 
b1 = int(samples*RATIO)
list_train = list_images_path[0:b1]
list_val = list_images_path[b1:]

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=N_VIEWS):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return  [self.base_transforms(x) for i in range(self.n_views)]
    
contrast_transforms = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=size),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5,
                                                       contrast=0.5,
                                                       saturation=0.5,
                                                       hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

transform = ContrastiveTransformations(contrast_transforms, n_views=N_VIEWS)

class MyDataset:
    def __init__(self, path_images,transform=None):
        self.transform = transform
        self.path_images = path_images
        
    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        
        image = Image.open(self.path_images[idx])
        #image = np.array(image)
        
        if self.transform:
            return self.transform(image)
        else:
            
            data_transformer = transforms.Compose([transforms.Resize(size),
                                                   transforms.ToTensor()])
            return data_transformer(image)
        
unlabeled_data_train = MyDataset(path_images = list_train, transform=transform)
unlabeled_data_val = MyDataset(path_images = list_val, transform=transform)

if __name__ == "__main__":
    print('****************len dataframe*********')
    print('****these datasets are for supervised learning******')
    print(f'train_df:{len(train_df)}, val_df:{len(val_df)}')
 

    print(f'len train_ds:{len(train_ds)}, len val ds: {len(val_ds)}')
    x, y = next(iter(train_ds))
    print(x.size())
    print(y.size())
    print(y)
    print('******** unlabeled data for unsupervised training****')
    print(f'unlabeled_data_train:{len(unlabeled_data_train)}')
    print(f'unlabeled_data_val:{len(unlabeled_data_val)}')
    print(f'unlabeled train first pic of first sample:{unlabeled_data_train[0][0].size()}')
    print(f'unlabeled train second pic of first sample:{unlabeled_data_train[0][1].size()}')

