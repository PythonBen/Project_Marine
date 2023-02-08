import torch.nn as nn
import torchvision.models as models

model_names = ["alexnet", "squeezenet", "mobilenet", "mobilenet_v3",
               "resnet18", "resnet34", "efficientnetb1", "efficientnetb2",
               "shuffle_net_v2_X0", "shuffle_net_v2_X2",
               "RegNet_Y_400MF", "RegNet_Y_800MF",
               "mnasnet0_5", "MNASNet1_3"]


# model size parameters: https://pytorch.org/vision/stable/models.html
def get_model_generic(nb_classes=7, pretrained=False, N=None):
    model_name = model_names[N]

    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=nb_classes, bias=True)

    elif model_name == "squeezenet":
        model = models.squeezenet1_1(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=nb_classes, kernel_size=(1,1), stride=(1,1))

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=nb_classes, bias=True)

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=nb_classes, bias=True)

    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=nb_classes, bias=True)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=nb_classes, bias=True)

    elif model_name == "efficientnetb1":
        model = models.efficientnet_b1(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=nb_classes, bias=True)

    elif model_name == "efficientnetb2":
        model = models.efficientnet_b2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1408, out_features=nb_classes, bias=True)

    elif model_name == "shuffle_net_v2_X0":
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        model.fc = nn.Linear(in_features=1024, out_features=nb_classes, bias=True)

    elif model_name == "shuffle_net_v2_X2":
        model = models.shufflenet_v2_x2_0()
        model.fc = nn.Linear(in_features=2048, out_features=nb_classes, bias=True)

    elif model_name =="RegNet_Y_400MF":
        model = models.regnet_x_400mf(pretrained=pretrained)
        model.fc = nn.Linear(in_features=400, out_features=nb_classes, bias=True)

    elif model_name =="RegNet_Y_800MF":
        model = models.regnet_x_800mf(pretrained=pretrained)
        model.fc = nn.Linear(in_features=672, out_features=nb_classes, bias=True)

    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=pretrained)
        model.classifier = nn.Linear(in_features=1280, out_features=nb_classes, bias=True)

    elif model_name == "MNASNet1_3":
        model = models.mnasnet1_3(pretrained=pretrained)
        model.classifier = nn.Linear(in_features=1280, out_features=nb_classes, bias=True)

    else:
        raise ValueError("Model not defined")

    #print(model)
    return model, model_name

if __name__ =="__main__":

    get_model_generic(N=7)

