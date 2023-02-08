import torch
import torch.nn as nn
import torch.nn.functional as F


# exemple of a well known convolutional model construction
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # convolution part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4, padding=O)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3,stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # fully connected part
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, image):
        # get the batch size, channels, heigth and width             # for a cnn: no = floor((ni - f + 2p)/s + 1)   # ppoling layer floor((ni - f )/s + 1)
        # of the input batch of images                                                  # (227-11)/4 + 1 = 55
        # original size : (bs, 3, 227, 227)
        bs, c, h, w = image.size()
        x = F.relu(self.conv1(image))            # size (bs, 96, 55, 55)
        x = self.pool1(x)                        # size (bs,96,  27, 27)                    # (55-3)/2+1 = 27
        x = F.relu(self.conv2(x))                # size (bs,256,  23, 23)                   # (27-5+2*2)/1 + 1 = 27
        x = self.pool2(x)                        # size (bs, 256, 13, 13)                   # (27-3)/2 + 1  = 13
        x = F.relu(self.conv3(x))                # size (bs, 384, 13, 13)                   # (13-3+2*1)/1 + 1 = 13
        x = F.relu(self.conv4(x))                # size (bs, 384, 13, 13)                   # (13-3+2*1)/1 + 1 = 13
        x = F.relu(self.conv5(x))                # size (bs, 256, 13, 13)                   # (13-3+2*1)/1 + 1 = 13

        x = self.pool3(x)                        # size (bs, 256, 6, 6)                     # (13-3)/2 + 1 = 6


        x = x.view(bs, -1)                       # size (bs,9216)
        x = F.relu(self.fc1(x))                  # size (bs,4096)
        x = self.dropout1(x)                     # size (bs,4096)
        x = F.relu(self.fc2(x))                  # size (bs,4096)
        x = self.dropout(x)                      # size (bs,4096)
        x = F.relu(self.fc3(x))                 # size( bs, 1000)

        # fo each sample in the batch
        x = torch.softmax(x, axis=1)             # size (bs, 1000)
        return x



