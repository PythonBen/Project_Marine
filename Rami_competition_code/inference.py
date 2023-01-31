from fastai.vision.all import *
import re
import time
import argparse
import main_clean

def parsing():
    """
    construct the parser for the paths
    :return: Pathlib object:
    paths to saved model ,
    path to the folder data,
    path to the output file,
    path to the test file
    path to pkl dataframe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_result", type=str, default="/media/ben/Data_linux/RAMI/code/predictions/",
                        help="path to the output file")
    parser.add_argument("-i", "--path_input", type=str, default="/media/ben/Data_linux/RAMI/"
                                                            "rami_folder/home/rami/"
                                                            "rami_marine_dataset/chosen_validation_data",
                        help = "path to the images used for inference")
    parser.add_argument("-pm", "--path_model", type=str, default="/media/ben/Data_linux/RAMI/code/models_multi/",
                        help = "path to the saved model")
    parser.add_argument("-pr", "--path_root", type=str, default="/media/ben/Data_linux/RAMI/rami_marine_dataset/",
                        help="path to the training data")
    parser.add_argument("-pdat", "--path_dataframe", type=str, default="/media/ben/Data_linux/RAMI/code/dataframe_fastai/",
                        help="path to the pkl dataframe")
    args = parser.parse_args()

    path_models = Path(args.path_model)
    path_root = Path(args.path_root)
    path_output_file = Path(args.path_result)
    path_input = Path(args.path_input)
    path_dataframe = Path(args.path_dataframe)
    return path_models, path_root, path_output_file, path_input, path_dataframe

def define_model(path_root=None, size=None, df=None):
    """ define the fastai model and its datablock and dataloader
    :param path_root: path to images
    :param size: size of the images (we resize all the images)
    :return: a fastai learner and the list with classes
    """

    data = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           splitter=RandomSplitter(),
                           get_x=ColReader('image_path', pref=str(path_root)+os.path.sep),
                           get_y=ColReader('lbl'),
                           item_tfms = Resize(size),)

    dls = data.dataloaders(df, bs = BS)
    classes = dls.vocab
    f1_score = F1Score(average='macro')
    precision = Precision(average='macro')
    recall = Recall(average='macro')
    learner = vision_learner(dls, resnet18, metrics=[accuracy, f1_score,precision, recall])
    return learner, classes

if __name__ == "__main__":

    # parameters
    BS=6
    SIZE = (256,256)
    # get the paths
    path_models, path_root, path_output_file, path_input, path_dataframe = parsing()

    # load the dataframe
    if (path_dataframe/"df.pkl").exists():
        print("Dataframe already built and saved as df.pkl")
        df = pd.read_pickle(path_dataframe/"df.pkl")
    else:
        B1 = main_clean.BuildDataframe(path_root=path_root)
        df = B1.build_list_and_df()
        #df.to_pickle(path_saved_dataframe/"df.pkl")                      # to save the dataframe, uncomment

    # define fastai learner model and load it
    #print(f"size:{SIZE}")
    learner, classes = define_model(path_root, size=SIZE, df=df)
    if (path_models/"learn18_multi_stage1.pkl").exists():
        #learn18_inf = learn18.load(path_models/"learn18_multi_stage1")
        learn18_inf = learner.load(path_models/"learn18_stage1_im256x256_bs6")
    else:
        LR = 1e-3
        learner.fine_tune(3, base_lr=LR, freeze_epochs=3)
        learn18_inf = learner

    # make a list of images to be tested
    list_test_images = main_clean.make_list_for_testing(path_input)
    print(list_test_images)
    learn18_inf.show_results()
    plt.show()

    # extract the model, its parameters and wieghts
    net, params, weight_softmax = main_clean.prepare_cam(learner=learn18_inf)

    # write the inference in the output file
    main_clean.write_outputfile(path_output_file, list_test_images, learn18_inf, path_input, weight_softmax,classes)


