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
    BS_US_train = 8          #128
    BS_US_val = 8            #128
    BS_SU_train = 8          #4
    BS_SU_val = 8            #4
    EPOCHS_US = 8            #30
    EPOCHS_SU = 4
    TRAIN_US = False
    TRAIN_SU = True
    INFERENCE = True
    N_VIEWS = 2
    RATIO = 0.8
    REDUC_FACTOR = 0.01   # for a toy dataset
    LR_US = 3e-2   #5e-4
    LR_SU = 3e-4
    WEIGHT_DECAY_US = 1.0
    WEIGHT_DECAY_SU = 0.1
    H_image = 96
    W_image = 96
    Projector_res18 = '2048-2048-2048'    # for resnet18 or resnet34
    Projector_res50 = '8192-8192-8192' #for resnet50
    GAMMA = 1
    NUM_CLASSES=7
    ARCH="res18"
    LAMBD = 0.000051    # separation bewteen the parts of the loss function, defautl: 0.0051
    PLOTTING_SU = True
    PLOTTING_US = False
    SANITY_CHECK_US = False
    SANITY_CHECK_SU = False
