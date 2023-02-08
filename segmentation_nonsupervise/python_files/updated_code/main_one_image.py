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

# original paper: Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering, W Kim and A Kanezaki, and M Tanaka
# github: https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
# script to segment one input image
# the output size will be the same as the input image size

def define_paths(input_folder, output_folder):
    """
    function to define the path for the input image and the path for the result
    :param input_folder: Path to the input folder
    :param output_folder: Path to the output folder
    :return: Path objets for the input and result folders
    """

    path_data = Path('/media/ben/Data_linux/code/unsupervised_learning/dataset/')
    path_input = path_data / input_folder
    path_output = path_data / output_folder
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    return path_input, path_output

def userarguments():
    parser = argparse.ArgumentParser(description="Parameters for the unsupervised segmentation of one image")
    # models parameters
    parser.add_argument("-si", "--size", default=(240, 320), type=tuple, help="Image Size")
    parser.add_argument("-nc", "--nChannel", default=20, type=int,
                        help="number of channel for the feature pixel vector")
    parser.add_argument("--nconv", default=2, type=int, help="number of convolution layer in the network model")
    parser.add_argument("-lr", "--learning_rate", default=0.1, type=int, help="learning rate use")
    parser.add_argument("-it", "--iterations", default=12, type=int, help="number of epochs")
    parser.add_argument("-sc", "--stp_con", default=3, type=int, help="weight for the continuity loss")
    parser.add_argument("-sm", "--stp_sim", default=1, type=int, help="weight for the similariy loss")
    # boolean parameters to save,plot,write or not
    parser.add_argument("-ml", "--min_labels", default=4, type=bool, help="desired final number of labels (classes)")
    parser.add_argument("--sanity_check", default=False, type=bool, help="make a sanity check or not")
    parser.add_argument("--save_model", default=True, type=bool, help="saving the model or not")
    parser.add_argument("--write_pics", default=True, type=bool, help="writing the segmented image output")
    parser.add_argument("--plot", default=True, type=bool, help="plot the training loss or not")
    parser.add_argument("--train", default=True, type=bool, help="training phase or not")
    parser.add_argument("--inference", default=True, type=bool, help="inference phase or not")
    parser.add_argument('--visualize', default=True, type=bool, help='visualization flag')
    parser.add_argument("--write_image", default=True, type=bool, help="saving and writing output segmented image")
    # folders names 
    parser.add_argument("--path_input", default="/mnt/narval/narval_BL/submeeting_2022/unsupervised_segmentation/one_image/Image_input/", type=str,
                        help="folder containing the input image")
    #parser.add_argument("--path_output", default="/mnt/narval/narval_BL/submeeting_2022/unsupervised_segmentation/one_imagev/Image_output/", type=str,
    #                    help="folder containing output image")
    parser.add_argument("--input_filename", default="input_image.jpg", type=str, help="input image")
    parser.add_argument("--output_filename", default="output_image.jpg", type=str, help="output image")

    return parser.parse_args()

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
    :param device: cpu or gpu
    :param nChannel: number of dimension of the pixel feature
    :return: similarity and continuity loss, Target in the vertical and horizontal direction
    """

    loss_fn = torch.nn.CrossEntropyLoss()

    loss_hpy = torch.nn.L1Loss(reduction='mean')
    loss_hpz = torch.nn.L1Loss(reduction='mean')

    HPy_target = torch.zeros(x.shape[0]-1, x.shape[1], nChannel)
    HPz_target = torch.zeros(x.shape[0], x.shape[1]-1, nChannel)
    HPy_target = HPy_target.to(device)
    HPz_target = HPz_target.to(device)

    return loss_fn, loss_hpy, loss_hpz, HPy_target, HPz_target


def training(model, iterations, lr, stepsize_con, stepsize_sim, size, nConv, nChannel, cwd):
    """ function to train on one image"""
    # define colours for vizualisation
    l1 = [0, 64, 128, 192, 255]
    p = product(l1, repeat=3)                      # there will be 5*5*5 = 125 different colours
    label_colours = np.array(list(p))
    loss_list = []

    for it in range(iterations):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1,2,0).contiguous().view(-1, nChannel)
        outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if VISUALIZE and TRAIN:
            im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imshow( "output", im_target_rgb )
            cv2.waitKey(20)

        # loss
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        print (it, '/', iterations, '|', ' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= minlabels:
            print ("nLabels", nLabels, "reached minLabels", minlabels)
            break

    # save output image
    if not VISUALIZE:
        output = model(data)[0]
        output = output.permute(1,2,0).contiguous().view(-1,nChannel)
        ignore, target = torch.max(output,1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    if WRITE:
        #cv2.imwrite(PATH_OUTPUT + f"iters_{it}_LR_{lr}_image_step_con_"
        #                       f"{stepsize_con}_step_sim_{stepsize_sim}_height_"
        #                      f"{size[0]}_width_{size[1]}_nconv_"
        #                          f"{nConv}_nchannel_{nChannel}_output.png", im_target_rgb)
        cv2.imwrite(PATH_OUTPUT.as_posix() + '/' +  filename_output, im_target_rgb)

    if SAVE_MODEL and TRAIN:
        torch.save(model.state_dict(), cwd / f"models/model_one_image_iter_{it}.pth")

    return loss_list


def inference(input_path, filename_input, output_path, filename_output, model, cwd):
    print(f'inference for the {filename_input}')
    print(f"The result will be located in {output_path} with the name:{filename_output}")
    l1 = [0, 64, 128, 192, 255]
    p = product(l1, repeat=3)                      # there will be 5*5*5 = 125 different colours
    label_colours = np.array(list(p))
    im = cv2.imread(input_path + filename_input)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    data = data.to(device)
    model.load_state_dict(torch.load(cwd/ f"models/model_one_image.pth"))
    model.eval()
    output = model(data)[0]
    output = output.permute(1,2,0).contiguous().view(-1,nChannel)
    ignore, target = torch.max(output,1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
    cv2.imwrite(output_path + filename_output, im_target_rgb)


if __name__ =="__main__":
    # parameters
    cwd = Path.cwd()
    
    (cwd/"models").mkdir(parents=True, exist_ok=True)
    args = userarguments()
    filename_input = args.input_filename
    filename_output = args.output_filename
    PATH_INPUT = args.path_input
    PATH_OUTPUT = cwd/"Image_output"
    PATH_OUTPUT.mkdir(parents=True, exist_ok=True)
    SIZE = args.size
    nConv = args.nconv
    minlabels = args.min_labels
    stepsize_con = args.stp_con
    stepsize_sim = args.stp_sim
    iterations = args.iterations
    lr = args.learning_rate
    nChannel = args.nChannel
    VISUALIZE = args.visualize
    TRAIN = args.train
    WRITE = args.write_image
    INFERENCE = args.inference
    PLOT = args.plot
    SAVE_MODEL = args.save_model
    print(f"size:{SIZE}")
    print(f"path_input:{PATH_INPUT}")

    # device
    if torch.cuda.is_available(): device = "cuda:0"
    else: device = "cpu"

    # load image
    im = cv2.imread(PATH_INPUT + filename_input)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    data = data.to(device)

    # model
    model = MyNet(input_dim=3, nChannel=nChannel, nConv=2).to(device)

    # loss functions and optimizer
    loss_fn, loss_hpy, loss_hpz, HPy_target, HPz_target = losses(im, device, nChannel)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if TRAIN:
        loss_list = training(model=model,
                             iterations=iterations,
                             lr=lr,
                             stepsize_con=stepsize_con,
                             stepsize_sim=stepsize_sim,
                             size=SIZE,
                             nConv=nConv,
                             nChannel=nChannel,
                             cwd=cwd)

    # inference
    if INFERENCE and (not TRAIN):
        inference(input_path=PATH_INPUT,
                  filename_input=filename_input,
                  output_path=PATH_OUTPUT,
                  filename_output=filename_output,
                  model=model,
                  cwd=cwd)
    if PLOT and TRAIN:
        plt.plot(range(len(loss_list)), loss_list)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.show()


