import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import argparse
import pathlib
from pathlib import Path
from get_models import get_model_generic
import engine
from dataset import ClassificationDataset_fromDataframe
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
import warnings

# inspired by the book: Approaching (almost) any machine learning probleme, Abhishek thakur.

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # classes
    # classes = ['algen', 'divers', 'robots', 'rocks', 'sand', 'structure', 'water_only']

    # models name
    # model_names = ["alexnet", "squeezenet", "mobilenet", "mobilenet_v3",
               # "resnet18", "resnet34", "efficientnetb1", "efficientnetb2",
               # "shuffle_net_v2_X0", "shuffle_net_v2_X2","RegNet_Y_400MF",
               #  "RegNet_Y_800MF", "mnasnet0_5", "MNASNet1_3"]

    # user choices
    TRAIN = False
    SAVE_MODEL = False
    INFERENCE = True
    WRITE_RESULT = True

    cwd = Path.cwd()

    #path_dataframe = Path('/mnt/narval/narval_BL/submeeting_2022/pickle_files')
    dataframe_name = "df_total_encoded_su.pkl"
    path_images = Path('/mnt/narval/narval_BL/submeeting_2022/datasets/dataset_su/total_320x240_su/')
    path_models = Path('/mnt/narval/narval_BL/submeeting_2022/supervised_model/')
    path_results = cwd/"results"
    path_results.mkdir(parents=True, exist_ok=True)
    path_dataframe = cwd/"dataframes"
    path_dataframe.mkdir(parents=True, exist_ok=True)
    def user_arguments():
        parser = argparse.ArgumentParser(description='Classification multi label avec pytorch')
        parser.add_argument("--path_images", default=path_images, type=pathlib.PosixPath, help="path to the image folder")
        parser.add_argument("--path_dataframe", default=path_dataframe, type=pathlib.PosixPath, help="path to the dataframe folder")
        parser.add_argument("--df_name", default=dataframe_name, type=str, help="dataframe name")
        parser.add_argument("--batch_size", default=16, type=int, help="batch size")
        parser.add_argument("--epochs", default=16, type=int, help="number of epochs")
        parser.add_argument("--size", default=(128, 128), type=tuple, help="image size")
        parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
        parser.add_argument("--size_crop", default=(96, 96), type=tuple, help="size_crop")

        return parser.parse_args()

    args = user_arguments()
    PRETRAINED = True
    Model_number = 3                         # according to the model name list

    model, model_type = get_model_generic(pretrained=PRETRAINED, N=Model_number)

    model.to(device)

    # parameters
    mean = (0.485, 0.456, 0.406)                         # imagenet statistics
    std = (0.229, 0.224, 0.225)

    BS = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    SIZE = args.size
    SIZE_CROP = args.size_crop

    # augmentations
    aug = A.Compose([A.HorizontalFlip(p=0.5),
                     A.VerticalFlip(p=0.5),
                     A.RandomCrop(*SIZE_CROP),
                     A.RandomBrightnessContrast(p=0.2),
                     A.RandomRotate90(p=0.5),
                     A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    aug2 = A.Compose([A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.RandomRotate90(),
                      A.RandomCrop(*SIZE_CROP),
                      A.GaussNoise(p=0.5),
                      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                      A.RandomBrightnessContrast(p=0.2),
                      A.ColorJitter(p=0.5),
                      A.Blur(blur_limit=3,p=0.5,),
                      A.RandomRotate90(p=0.2),
                      A.HueSaturationValue(p=0.5),
                      A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    norm  = A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

    # get the data
    df = pd.read_pickle(args.path_dataframe/args.df_name)

    # split the dataframe, train and valid
    df_train, df_val = train_test_split(df, test_size=0.2)
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    # augmentation on the training dataset
    ds_train = ClassificationDataset_fromDataframe(image_paths=args.path_images,
                                                   df=df_train,
                                                   resize=SIZE,
                                                   augmentations=aug2)

    # no augmentation on the valid dataset
    ds_valid = ClassificationDataset_fromDataframe(image_paths=args.path_images,
                                                   df=df_val,
                                                   resize=SIZE,
                                                   augmentations=norm)

    # create train and valid loaders
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BS, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=BS, shuffle=False, num_workers=4)

    print(f'len train loader:{len(train_loader)}, valid loader:{len(valid_loader)}')

    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def train_epochs(epochs=None, model=model, lr=LR):

        # optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

        train_loss_list = []
        valid_loss_list = []
        accuracy_list = []
        recall_list = []
        precision_list = []
        F1_list = []

        for epoch in range(epochs):
            loss_train = engine.train(train_loader, model, optimizer, device=device)
            scheduler.step()
            current_lr = get_lr(optimizer)
            print(f"current_lr:{current_lr}")
            loss_valid, accuracy, precision, recall, F1 = engine.evaluate(valid_loader, model, device=device)
            print(f"epoch:{epoch}, loss_train:{loss_train:.2f}, loss_valid:{loss_valid:.2f}")
            print(f"accuracy:{accuracy:.2f}, precision:{precision:.2f}, recall:{recall:.2f}, f1:{F1:.2f}")
            train_loss_list.append(loss_train)
            valid_loss_list.append(loss_valid)
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            precision_list.append(precision)
            F1_list.append(F1)

        return train_loss_list, valid_loss_list, accuracy_list, precision_list, recall_list, F1_list, model

    def plotting():

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(range(1, EPOCHS+1), train_loss_list, label='train')
        ax1.plot(range(1, EPOCHS+1), valid_loss_list, label='valid')
        ax1.set_title("Train and valid loss")
        ax1.legend()
        ax2.plot(range(1, EPOCHS+1), accuracy_list, label='accuracy')
        ax2.plot(range(1, EPOCHS+1), precision_list, label='precision')
        ax2.plot(range(1, EPOCHS+1), recall_list, label='recall')
        ax2.legend()

        plt.xlabel("epochs")
        plt.ylabel("metrics")
        plt.legend()
        plt.show()

    if TRAIN:
        train_loss_list, valid_loss_list, accuracy_list, precision_list, recall_list, F1_list, model =  train_epochs(epochs=EPOCHS)
        plotting()

    def load_model(model_name, model_number):

        path_to_model = path_models/model_name
        print(path_to_model)
        model, model_type = get_model_generic(pretrained=False, N=model_number)
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
        return model

    if INFERENCE:
        #model = load_model(model_name="resnet18_epochs_20_pretrained_True.pth", model_number=Model_number)
        model = load_model(model_name="mobilenet_v3_epochs_60_pretrained_True_6e-05.pth", model_number=Model_number)
        model.eval()
        model.to(device)
        loss_valid, accuracy, precision, recall, F1 = engine.evaluate(valid_loader, model, device=device)
        print(f"accuracy:{accuracy:.2f}")
        print(f"precision:{precision:.2f}")
        print(f"recall:{recall:.2f}")
        print(f"f1:{F1:.2f}")
        checkf1 = 2*precision*recall/(precision+recall)
        print(f"check:{checkf1}")

# save the dict model weights for inference use later
    def save_model():
        model_name = f"{model_type}_epochs_{EPOCHS}_pretrained_{PRETRAINED}_{LR}.pth"
        path_to_model = path_models/model_name
        torch.save(model.state_dict(), path_models/model_name)
        return path_to_model
    if SAVE_MODEL:
        path_to_model = save_model()

    def inference_image_to_class(nb=2, model=model, thresh=0.5):

        count=0
        for step, data in enumerate(valid_loader):
            print(f"count:{count}")
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.detach().cpu().numpy()
            predicted_classes = ((model(inputs)).sigmoid() > thresh).short()
            predicted_classes = predicted_classes.detach().cpu().numpy()
            print('predicted_classes:')
            print(type(predicted_classes))
            print(predicted_classes)
            print('***************')
            print('labels')
            print(type(targets))
            print(targets)
            if WRITE_RESULT:
                with open(path_results/f"batch_{count}.txt", 'w') as f:
                    f.write(f"Overall accuracy on the valid dataset:{accuracy:.2f}\n")
                    f.write(f" Overall precision:{precision:.2f}; Overall recall:{recall:.2f}; Overall f1:{F1:.2f}\n")
                    f.write("*"*10)
                    f.write("\n")
                    f.write("*"*10 + " predictions " + "*"*10)
                    f.write('\n\n')
                    f.write(str(predicted_classes))
                    f.write('\n')
                    f.write('\n')
                    f.write("*"*10 + " ground truth " + "*"*10)
                    f.write('\n\n')
                    f.write(str(targets))

            inputs = inputs.detach().cpu()
            imggrid = torchvision.utils.make_grid(inputs, nrow=4, normalize=True, pad_value=0.9)
            imggrid = imggrid.permute(1, 2, 0)
            plt.imshow(imggrid)
            path_saveim = path_results/f"batch_{count}.png"

            if WRITE_RESULT:
                plt.savefig(path_saveim)

            plt.show()
            count+=1
            if count==nb:
                break

    if INFERENCE:
        inference_image_to_class(nb=4, model=model)

