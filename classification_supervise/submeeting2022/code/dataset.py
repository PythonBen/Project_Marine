import pathlib

import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
# if have images without ending bits
ImageFile.LOAD_TRUNCATED_IMAGES = True

# path files specific to my computer
#path_dataframe = Path('/media/ben/Ubuntu 20_04_4 LTS amd64/submeeting_2022/submeeting_images/pickle_dataframe_saved/')
#dataframe_name = "df_total_encoded_su.pkl"
#path_images = Path('/media/ben/Ubuntu 20_04_4 LTS amd64/submeeting_2022/submeeting_images/total_images_su/total_320x240_su/')

#path_dataframe = Path('/mnt/narval/narval_BL/submeeting_2022/pickle_files')
#dataframe_name = "df_total_encoded_su.pkl"
#path_images = Path('/mnt/narval/narval_BL/submeeting_2022/datasets/dataset_su/total_320x240_su/')

#print(type(path_dataframe))
cwd = Path.cwd()

def userarguments():
    parser = argparse.ArgumentParser(description="Parameters for the supervised classification of images")

    # folders names
    #parser.add_argument("--path_dataframe", default=Path('/mnt/narval/narval_BL/submeeting_2022/pickle_files'),
     #                   type=pathlib.PosixPath,
     #                   help="path to the dataframe")
    parser.add_argument("--path_dataframe", default=cwd/"dataframes",
                        type=pathlib.PosixPath,
                        help="path to the dataframe")
    parser.add_argument("--path_images", default=Path('/mnt/narval/narval_BL/submeeting_2022/datasets/dataset_su/total_320x240_su/'),
                        type=pathlib.PosixPath,
                        help="path to the images folder")
    parser.add_argument("--dataframe", default="df_total_encoded_su.pkl", type=str,
                        help="dataframe name")

    return parser.parse_args()

args = userarguments()
path_dataframe = args.path_dataframe
dataframe_name = args.dataframe
path_images = args.path_images


# parameters
SIZE = (128, 128)
RATIO = 0.8

# General classification dataset with a dataframe"""

class ClassificationDataset_fromDataframe:

    def __init__(self,
                 image_paths,
                 df,
                 resize=None,
                 augmentations=None):
        """
        :param image_paths: path to image
        :param path_dataframe: path to the corresponding dataframe
        :param df_name: datarame name
        :param resize: tuple
        :param augmentations: albumentations augmentation
        """
        self.image_paths = image_paths
        self.df = df
        self.resize = resize
        self.augmentations = augmentations
        self.targets = self.df.loc[:, "labels_encoded"]

    def __len__(self):
        """ Return the total number of sample in the dataset"""
        return len(self.df)

    def __getitem__(self, item):
        """ For a given item, return the image array and its associated label.
         Needed for training a model """
        suffix = ".jpg"
        # use PIL to open an image
        image_path = self.image_paths/self.df.loc[item, "fname"]
        image_path = image_path.as_posix() + suffix
        image = Image.open(image_path)
        # convert Image to RGB, we have single channel images
        image = image.convert("RGB")
        # grab targets
        targets = self.targets[item]
        # resize if needed
        if self.resize is not None:
            image = image.resize((self.resize[0], self.resize[1]), resample=Image.BILINEAR)
        # convert image to numpy array
        image = np.array(image)
        # if we have albumentations augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        # pytorch expect CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # return tensors of image and targets
        # take a look at the type
        # for regression tasks, dtype of targets will change to torch.float
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }

if __name__ == "__main__":

    def check():
        df = pd.read_pickle(path_dataframe/dataframe_name)
        return df
    df = check()
    print(df.head())
    print(df.columns)
    print(df.shape)
    ds = ClassificationDataset_fromDataframe(image_paths=path_images,
                                             df=df,
                                             resize=SIZE,
                                             augmentations=None)


    ds_train, ds_valid = train_test_split(ds, test_size=0.2)
    print(f"len train:{len(ds_train)}, len valid:{len(ds_valid)}")

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=4)
    for item in ds_train:
        print(item['image'].shape)
        print(item['targets'].shape)
        print(item['targets'])
        break

    for item in train_loader:
        print(item["image"].shape)
        print(item["targets"].shape)
        break
