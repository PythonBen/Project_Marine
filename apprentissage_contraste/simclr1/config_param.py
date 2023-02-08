#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:29:11 2021

@author: lepersbe
"""
# configutation file
class Parameters:
    BS_US = 128
    BS_SU_train = 16
    BS_SU_val = 16
    EPOCHS_US = 100
    EPOCHS_SU = 4
    TRAIN_US = False
    TRAIN_SU = False
    N_VIEWS = 2
    RATIO = 0.8
    LR_US = 5e-4
    LR_SU = 1e-3
    TEMPERATURE = 0.07
    WEIGHT_DECAY_US = 1e-4
    WEIGHT_DECAY_SU = 1e-3
    HIDDEN_DIM = 128
    H_image = 128
    W_image = 128

