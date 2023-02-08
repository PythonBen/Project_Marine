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
    N=4
    BS_US = 64*N             
    BS_SU_train = 16
    BS_SU_val = 16
    EPOCHS_US = 20*N
    EPOCHS_SU = 4
    TRAIN_US = False
    TRAIN_SU = False
    N_VIEWS = 2
    RATIO = 0.8
    LR_US = 5e-4
    LR_SU = 1e-3
    WEIGHT_DECAY_US = 1e-4
    WEIGHT_DECAY_SU = 1e-3
    H_image = 128
    W_image = 128
    Projector = '2048-2048-2048'

