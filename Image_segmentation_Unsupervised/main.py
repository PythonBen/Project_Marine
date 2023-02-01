import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from itertools import product
import argparse

# unsupervised clustering: Paper: Unsupervised learning of image segmentation based on differential feature clustering, W, Kim and A, Kanezaki, and M, Tanaka
# github repo: https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
# we have modify the code by adding functions and the use of dataloaders for batch training




def make_list_image(path_train, path_valid, bs):
    """
    function to make lists of train and valid images
    :param path_train: Path to the train folder
    :param path_valid: Path to the valid folder
    :param bs: batch size parameter. The number of images in the train and valid folder should be a integer multiple of batch_size
    :return: list of images for the train and valid parts
    """

    train_images = [im.as_posix() for im in path_train.iterdir()]
    valid_images = [im.as_posix() for im in path_valid.iterdir()]
    if len(train_images) % bs != 0:
        print("The train images folder should contain a integer number of batch")
    if len(valid_images) % bs != 0:
        print("The valid images folder should contain a integer number of batch")
    return train_images, valid_images


# DatasetClass
class ImageDataset:
    """ Class to build the image dataset """

    def __init__(self, image_paths, size):
        self.image_paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        resized_im = cv2.resize(image, dsize=self.size)
        resized_im = resized_im.transpose((2, 0, 1)).astype('float32') / 255.

        return torch.tensor(resized_im)


def make_dataset_dataloader(train_images, valid_images, size, ds_and_dl=False):
    """
    function to build the dataloaders from the train and image list
    :param train_images: list of train images
    :param valid_images: list_of valid images
    :param size: size of the resize image
    :return: train, valid dataloaders, train and valid datasets
    """

    ds_train = ImageDataset(image_paths=train_images, size=size)
    ds_valid = ImageDataset(image_paths=valid_images, size=size)
    train_dl = torch.utils.data.DataLoader(dataset=ds_train, batch_size=BS, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(dataset=ds_valid, batch_size=BS, shuffle=False)
    if ds_and_dl:
        print(f"len(ds_train):{len(ds_train)}, len(ds_valid):{len(ds_valid)}")
        print(f"len(train_dl):{len(train_dl)}, len(valid_dl):{len(valid_dl)}")
    return train_dl, valid_dl, ds_train, ds_valid


# define a cnn  model
class MyNet(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super(MyNet, self).__init__()
        self.input_dim = input_dim
        self.nChannel = nChannel
        self.nConv = nConv

        self.conv1 = nn.Conv2d(self.input_dim, self.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.nConv - 1):
            self.conv2.append(nn.Conv2d(self.nChannel, self.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(self.nChannel))
        self.conv3 = nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


def losses(x, device, nChannel):
    """
    function to compute the losses: continuity loss in the vertical and horizontal direction,
    similarity loss (crossentropy)
    :param x: a tensor from the train dataloader
    :return: similarity and continuity loss, Target in the vertical and horizontal direction
    """

    loss_fn = torch.nn.CrossEntropyLoss()

    loss_hpy = torch.nn.L1Loss(reduction='mean')
    loss_hpz = torch.nn.L1Loss(reduction='mean')

    HPy_target = torch.zeros(x.shape[0], x.shape[2] - 1, x.shape[3], nChannel)
    HPz_target = torch.zeros(x.shape[0], x.shape[2], x.shape[3] - 1, nChannel)

    HPy_target = HPy_target.to(device)
    HPz_target = HPz_target.to(device)

    return loss_fn, loss_hpy, loss_hpz, HPy_target, HPz_target


def loss_one_epoch(model, dataloader, sanity_check, lr, stepsize_con, stepsize_sim, min_labels, print_labels=False):
    """
    function to train on one epoch
    :param model: pytorch model
    :param dataloader: dataloader (pytorch tensors of images)
    :param sanity_check: boolean for checking
    :param lr: learning_rate
    :param stepsize_con: weight for the continuity loss
    :param stepsize_sim: weight for the similarity loss
    :param min_labels: minimum number of classes
    :return: average loss on the whole train loader, boolean min label indicator
    """

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    running_loss = 0.0
    bool_minlabel = False
    for image in dataloader:
        image = image.to(device)
        optimizer.zero_grad()
        output = model(image)
        output = output.permute(0, 2, 3, 1).contiguous().view(BS, -1, nChannel)
        outputHP = output.reshape((BS, image.shape[2], image.shape[3], nChannel))
        HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
        HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]

        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        output = output.reshape(output.shape[0] * output.shape[1], -1)
        ignore, target = torch.max(output, 1)
        nLabels = len(torch.unique(target))
        if print_labels:
            print(f"nlabels:{nLabels}")
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if sanity_check:
            break
        if nLabels <= min_labels:
            print("nLabels", nLabels, "reached minLabels", MIN_LABELS, ".")
            bool_minlabel = True
            break

    return running_loss / len(dataloader), bool_minlabel


def training_loop(epochs, model, dl, sanity_check, lr, stepsize_con, stepsize_sim, min_labels):
    """
    function to train the model
    :param epochs: epochs
    :param model: pytorch model
    :param dl: data loader
    :param sanity_check: bool to check if the training loop works
    :param lr: learning rate
    :param stepsize_con: weight of the continuity function
    :param stepsize_sim: weight of the similarity function
    :param min_labels: number of class desired
    :return: model and loss history
    """

    loss_history = []
    for epoch in range(epochs):
        model.train()
        loss_epoch, min_label_reached = loss_one_epoch(model=model,
                                                       dataloader=dl,
                                                       sanity_check=sanity_check,
                                                       lr=lr,
                                                       stepsize_con=stepsize_con,
                                                       stepsize_sim=stepsize_sim,
                                                       min_labels=min_labels)
        loss_history.append(loss_epoch)
        print(f"epoch:{epoch}, loss:{loss_epoch}")
        if min_label_reached:
            print("minimum labels reached")
            break
    return M1, loss_history


def inference(data_loader, file_path_saved):
    """
    function for inference
    :param data_loader: data loader from the valid set
    :param file_path_saved: path to saved the results (segmented images)
    :return: list of segmented images
    """

    list_im = []
    np.random.seed(42)
    # label_colours = np.random.randint(255, size=(100, 3))
    # p = product(range(0,260,60),repeat=3)
    # label_colours = np.array(list(p))
    # l1 = [0, 64, 128, 255]
    # l1 = [0,128,255]
    l1 = [0, 64, 128, 192, 255]
    p = product(l1, repeat=3)
    label_colours = np.array(list(p))
    for bn, img in enumerate(data_loader):
        M1.eval()
        img = img.to(device)
        output = M1(img)
        output = output.permute(0, 2, 3, 1).view(BS, -1, nChannel)
        ignore, target = torch.max(output, 2)
        inds = target.data.cpu().numpy().reshape((BS, img.shape[2], img.shape[3]))
        inds_rgb = np.array([label_colours[c % nChannel] for c in inds])
        inds_rgb = inds_rgb.reshape(img.shape).astype(np.uint8)

        print(f"batch number:{bn}")
        if WRITE_PICS:

            len_valid = len(valid_dl)
            iter_list = list(path_valid.iterdir())
            for i in range(BS):
                filename = iter_list[bn * BS + i].name
                output_im = inds_rgb[i, :, :, :]
                output_im2 = np.reshape(output_im, (*SIZE, 3))
                list_im.append(output_im2)
                cv2.imwrite(file_path_saved.as_posix() + '/' + f"{filename}", output_im2)

    return list_im


def userarguments():
    parser = argparse.ArgumentParser(description="Parameters for the unsupervised segmentation of images in batch")
    # models parameters
    parser.add_argument("-si", "--size", default=(128, 128), type=tuple, help="Image Size")
    parser.add_argument("-bs", "--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("-nc", "--nchannel", default=60, type=int,
                        help="number of channel for the feature pixel vector")
    parser.add_argument("--nconv", default=2, type=int, help="number of convolution layer in the network model")
    parser.add_argument("-lr", "--learning_rate", default=0.1, type=int, help="learning rate use")
    parser.add_argument("-ep", "--epochs", default=5, type=int, help="number of epochs")
    parser.add_argument("-sc", "--stp_con", default=5, type=int, help="weight for the continuity loss")
    parser.add_argument("-sm", "--stp_sim", default=1, type=int, help="weight for the similariy loss")
    # boolean parameters to save,plot,write or not
    parser.add_argument("-ml", "--min_labels", default=4, type=bool, help="desired final number of labels (classes)")
    parser.add_argument("--sanity_check", default=False, type=bool, help="make a sanity check or not")
    parser.add_argument("--save_model", default=False, type=bool, help="saving the model or not")
    parser.add_argument("--write_pics", default=True, type=bool, help="writing the segmented image output")
    parser.add_argument("--plot", default=True, type=bool, help="plot the training loss or not")
    parser.add_argument("--train", default=True, type=bool, help="training phase or not")
    parser.add_argument("--inference", default=True, type=bool, help="inference phase or not")
    # folders names
    parser.add_argument("--path_root", default="/media/ben/Data_linux/code/unsupervised_learning/dataset/", type=str,
                        help="parent folder")
    parser.add_argument("-tf", "--train_folder", default="ref_sub", type=str,
                        help="folder containing the training images")
    parser.add_argument("-vf", "--valid_folder", default="valid_sub", type=str,
                        help="folder containing the valid images")
    parser.add_argument("-rf", "--result_folder", default="result_segmentation_submeeting", type=str,
                        help="folder containing the result of the segmentation")

    return parser.parse_args()


if __name__ == "__main__":

    args = userarguments()
    # interesting parameters : lr = 0.04, nChannel=60, STEPSIZE_CON = 5
    cwd = Path.cwd()
    print(f"cwd:{cwd}")

    # parameters
    SIZE = args.size
    BS = args.batch_size
    nChannel = args.nchannel
    nConv = args.nconv
    LR = args.learning_rate
    EPOCHS = args.epochs
    SANITY_CHECK = args.sanity_check
    STEPSIZE_CON = args.stp_con
    STEPSIZE_SIM = args.stp_sim
    SAVE_MODEL = args.save_model
    WRITE_PICS = args.write_pics
    PLOT = args.plot
    MIN_LABELS = args.min_labels
    TRAIN = args.train
    INFERENCE = args.inference
 
    PATH_ROOT = Path(args.path_root)
    TRAIN_FOLDER = Path(args.train_folder)
    VALID_FOLDER = Path(args.valid_folder)
    RESULT_FOLDER = Path(args.result_folder)
    
    # path data
                                                                  
    path_train = PATH_ROOT / TRAIN_FOLDER
    path_valid = PATH_ROOT / VALID_FOLDER
    path_result = PATH_ROOT /  RESULT_FOLDER
    train_images, valid_images = make_list_image(path_train, path_valid, BS)
    print(f"list_im:{len(train_images)}, val_im:{len(valid_images)}")

    # device
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # dataloaders
    train_dl, valid_dl, ds_train, ds_valid = make_dataset_dataloader(train_images=train_images,
                                                                     valid_images=valid_images,
                                                                     size=SIZE)

    # model
    M1 = MyNet(input_dim=3, nChannel=nChannel, nConv=nConv).to(device)

    # loss functions
    for item in train_dl:
        x = item
        break
    print(f"x.size:{x.size()}")
    # loss functions and optimizer
    loss_fn, loss_hpy, loss_hpz, HPy_target, HPz_target = losses(x, device, nChannel)
    optimizer = torch.optim.SGD(M1.parameters(), lr=LR, momentum=0.9)

    if TRAIN:
        M1, loss_history = training_loop(epochs=EPOCHS,
                                         model=M1,
                                         dl=train_dl,
                                         sanity_check=SANITY_CHECK,
                                         lr=LR,
                                         stepsize_con=STEPSIZE_CON,
                                         stepsize_sim=STEPSIZE_SIM,
                                         min_labels=MIN_LABELS)
    else:
        # loading a train model
        M1 = MyNet(input_dim=3, nChannel=nChannel, nConv=nConv).to(device)
        M1.load_state_dict(torch.load(cwd / f"models/model_batch_{SIZE[0]}.pth"))
        M1.eval()
    if SAVE_MODEL:
        torch.save(M1.state_dict(), cwd / f"models/model_batch_{SIZE[0]}.pth")

    # training loss
    if PLOT and TRAIN:
        list_epochs = range(len(loss_history))
        plt.plot(list_epochs, loss_history)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

    # folder to save the results, inference
    if INFERENCE:
        folder_saved = path_result / f"EPOCHS_{EPOCHS}_LR_{LR}_image_step_con_{STEPSIZE_CON}_step_sim_{STEPSIZE_SIM}_height_{SIZE[0]}_width_{SIZE[1]}_nconv_{nConv}_nchannel_{nChannel}/"
        if not os.path.exists(folder_saved):
            os.mkdir(folder_saved)

        list_im = inference(data_loader=valid_dl, file_path_saved=folder_saved)
