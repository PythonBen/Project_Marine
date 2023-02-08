#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:29:11 2021

@author: lepersbe
"""
"""
# configutation file
BS : Batch Size
US: Unsupervised, SU: Supervised
N_VIEWS: number of augmentations for 1 images, here it is 2
RATIO: split Ratio to make the training and valid set
LR: learning rate
Projector: architecture of the projector network
"""
class Parameters:
    N=1
    BS_US_train = 512*2          #128
    BS_US_val = 512*2            #128
    BS_SU_train = 2*4          #4
    BS_SU_val = 2*4            #4
    EPOCHS_US = 1000            #30
    EPOCHS_SU = 4
    N_VIEWS = 2              # # of augmentations
    RATIO = 0.8
    LR_US = 5e-4   #5e-4
    LR_SU = 5e-2
    WEIGHT_DECAY_US = 1e-4
    WEIGHT_DECAY_SU = 1.0
    HIDDEN_DIM = 128
    H_image = 128  #96
    W_image = 128  #96
    Projector_res18 = '2048-2048-2048'    # for resnet18 or resnet34
    Projector_res50 = '8192-8192-8192' #for resnet50
    TRAIN_US = False
    TRAIN_SU = False
    INFERENCE = True
    TEMPERATURE = 0.07
    NUM_CLASSES = 7
    GAMMA = 0.9      # learning rate decay
# to do try with larger image ?
# perform inference

