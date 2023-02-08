"""
Created on Mon Nov  8 16:15:28 2021

@author: lepersbe
# file to create the datasets for unsupervised training (no labels)
and supervised training (with labels)
For the labels, we use one hot encoding to encode the 7 classes : cl = ['algen', 'divers', 'robots', 'rocks', 'sand', 'structure','water_only']
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


#path_root = Path('/media/ben/Data_linux/code/submeeting_code/')
path_root = Path('/mnt/narval/narval_BL/submeeting_2022/')

path_data = path_root/'datasets'
path_data_su = path_data/'dataset_su'
path_data_us = path_data/'dataset_us'
# structure of the directory  code - datasets - pickle_files
# inside the datasets folder: dataset_su - dataset_us       (supervised and unsupervised)
pickle_dataframe = 'pickle_files'
path_dataframe = path_root/pickle_dataframe

dataframe = 'df_total_datasetsu.pkl'
folders = ['boulouris2', 'pyramide2', 'pyramide3', 'pyramide4']
frames_range = ['0_3525', '0_900', '0_380', '0_900']
folder_total_su = 'total_320x240_su'
# Parameters
P = Parameters()
BS_US_train = P.BS_US_train
BS_US_val = P.BS_US_val
BS_SU_train = P.BS_SU_train
BS_SU_val = P.BS_SU_val
EPOCHS_US = P.EPOCHS_US
EPOCH_SU = P.EPOCHS_SU
N_VIEWS = P.N_VIEWS
RATIO = P.RATIO
H, W  = P.H_image, P.W_image
size = (H, W)

im_size = ['_320x240_SU', '_640x480_SU']
SIZE = (320,240)
N=4
df = pd.read_pickle(path_dataframe/dataframe)

print(f'path_data:{path_data}')
print(df.head())

# ****************************************************
# make a dataset with labels for Supervised learning
class SupervisedDataset(Dataset):

    def __init__(self, dataframe, path_data, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform
        self.path_data = path_data

    def __len__(self):
        return len(self.dataframe)

    def get_x(self,i):
        """ extract the dependent variable (image) """

        im = (self.dataframe.iloc[i]['fname'])
        im_full = path_data_su/folder_total_su/im
        #print(f'im_full:\n{im_full}')
        return im_full

    def get_y(self,i):
        return torch.tensor(self.dataframe.iloc[i]['labels_encoded'])

    def __getitem__(self,idx):

        image = Image.open(self.get_x(idx)).convert('RGB') #PIL image
        label = self.get_y(idx)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

Sudat = SupervisedDataset(dataframe=df, path_data=path_data)

def check():
    im1, lb1 = Sudat[300]
    print(im1.size)
    print(lb1)
    print(len(Sudat))
check()
transform_basic = transforms.Compose([transforms.Resize(SIZE),transforms.ToTensor()])

# dataloaders
def splitter(df):
    ''' create the train and valid indices according to the boolean "is_valid" column'''
    train = df[df['is_valid']==False]
    valid = df[df['is_valid']==True]
    return train, valid

train_df, val_df = splitter(df)
train_ds = SupervisedDataset(train_df, path_data, transform=transform_basic)
val_ds = SupervisedDataset(val_df, path_data, transform=transform_basic)

# make a unlabeled dataset for unsupersvised learning

im_size_us = ['_320x240_US', '_640x480_US']                   # boulouris2_frames_0_3525_320x240_US

folders_frames = list(zip(folders, frames_range))

list_folder_us = []
for i in range(len(folders_frames)):
    list_folder_us.append(folders_frames[i][0] + '_frames_' + folders_frames[i][1] + im_size_us[0])

list_paths_images = [path_data_us/folders_frames[i][0]/list_folder_us[i] for i in range(len(folders_frames))]


def counting_images(n):
    count = 0
    for item in list_paths_images[n].iterdir():
        count+=1
    return count

def counting_in_folder():
    total = 0
    for i in range(len(list_paths_images)):
        count = counting_images(i)
        total += count
        folder_name = list_paths_images[i].parent.parts[-1]
        print(f"{count} images in {folder_name}")
    print(f"{total} in total")

# make a list with all the images:
list_images_total = []
for folder in list_paths_images:
    for image in folder.iterdir():
        list_images_total.append(image)

# *************************
# make a unlabeled dataset for unsupersvised learning
#counting_in_folder()
RATIO = 0.8
reduc_factor = 0.2
samples = int(len(list_images_total)*reduc_factor)
#print(f'samples:{samples}')
list_images_reduc = list_images_total[0:samples]

b1 = int(samples*RATIO)
list_train = list_images_reduc[0:b1]
list_val = list_images_reduc[b1:]


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
