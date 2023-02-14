import pathlib

from fastai.vision.all import *
import argparse

def userarguments():
    parser = argparse.ArgumentParser(description="Parameters for the multilabel classification with fastai")
    parser.add_argument("--path_root", default=Path("/mnt/narval/narval_BL/submeeting_2022/"), type=pathlib.PosixPath, help="Image Size")

    return parser.parse_args()

def paths():
    args = userarguments()
    path_root = args.path_root
    path_dataset = path_root/"datasets/dataset_su/total_320x240_su"
    path_dataframe = path_root/"pickle_files/df_total_datasetsu.pkl"
    path_pickle = path_root/'pickle_dataframes/'
    return path_root, path_dataset, path_dataframe, path_pickle

path_root, path_dataset, path_dataframe, path_pickle = paths()

class DataDistribution:
    """ create a dictionary with class and plot an histogram"""
    def __init__(self, df):
        self.df = df

    def data_dict(self):
        """ fonction to return a dictionnary with the number of images per classes"""

        classes = ['algen', 'divers', 'robots', 'rocks', 'sand', 'structure', 'water_only']

        algen_rocks = df.loc[df.labels=='algen rocks', 'labels'].count() #03
        algen_sand = df.loc[df.labels=='algen sand', 'labels'].count()   #04
        algen_rocks_sand = df.loc[df.labels=='algen rocks sand', 'labels'].count() #034
        algen_rocks_sand_structure = df.loc[df.labels=='algen rocks sand structure', 'labels'].count() #0345
        algen_sand_structure = df.loc[df.labels=='algen sand structure', 'labels'].count() #045
        algen_rocks_structure = df.loc[df.labels=='algen rocks structure', 'labels'].count()  #035
        algen_rocks_robots = df.loc[df.labels=='algen rocks robots', 'labels'].count() #023
        algen_rocks_robots_structure = df.loc[df.labels=='algen rocks robots structure', 'labels'].count() #0235
        algen_divers_rocks = df.loc[df.labels=='algen divers rocks', 'labels'].count()       #013
        algen_rocks_robots_sand = df.loc[df.labels=='algen rocks robots sand', 'labels'].count() #0234
        water = df.loc[df.labels=='water_only', 'labels'].count()   #6
        robots_structure = df.loc[df.labels=='robots structure', 'labels'].count() #25
        divers = df.loc[df.labels=='divers', 'labels'].count() #1
        robots = df.loc[df.labels=='robots', 'labels'].count() #2

        dict_multiclass = {'03':algen_rocks,
                       '04':algen_sand,
                       '034':algen_rocks_sand,
                       '0345':algen_rocks_sand_structure,
                       '045':algen_sand_structure,
                       '035':algen_rocks_structure,
                       '023':algen_rocks_robots,
                       '0235':algen_rocks_robots_structure,
                       '013': algen_divers_rocks,
                       '0234': algen_rocks_robots_sand,
                        '6': water,
                        '25': robots_structure,
                        '1': divers,
                        '2': robots
                            }
        return dict_multiclass

    def plot_hist(self):
        dfhist = pd.Series(self.data_dict(), index =['03', '04', '034', '0345',
                                            '045', '035','023','0235','013','0234','6', '25', '1', '2'])
        dfhist.plot.bar()
        plt.xlabel('class')
        plt.ylabel('number')

if __name__ == "__main__":
    PLOT_DATA = False
    SHOW_BATCH = False
    TRAIN = False
    INFERENCE = True
    # 1. Dataframe
    df_new = pd.read_pickle(path_dataframe)
    df = df_new.drop_duplicates(subset=['fname']).reset_index(drop=True)
    df = df.drop(columns='labels_encoded', axis=1)               # fastai will encod  automatically the class labels
    print(df.head())

    # 2. Data
    if PLOT_DATA:
        D1 = DataDistribution(df)
        print(D1.data_dict())
        D1.plot_hist()
        plt.show()

    def check():
        D1 = DataDistribution(df)
        dict_multiclass = D1.data_dict()
        total = sum(dict_multiclass.values())
        print(f"total number of images: {total}")

        assert(total == df.shape[0])
# check()

    # 3. functions required to build the datablock class
    def get_x(r): return path_dataset/r['fname']                    # acess the  dependent X variable (image)
    def get_y(r): return r['labels'].split(' ')                     # acess the independent Y variable (labels)
    def splitter(df):                                               # define how we split the train and valid dataset
        train = df.index[~df['is_valid']].tolist()
        valid = df.index[df['is_valid']].tolist()
        return train, valid

    train, valid = splitter(df)
    print(f'train dataset size: {len(train)}, valid dataset size: {len(valid)}')

    # 4. Parameters and metrics
    # Parameters
    IM_SIZE = (128,128)               # image size
    BS= 8                             # batch size
    LR = 3e-3                         # learning rate

    # metrics
    f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
    f1_macro.name = 'F1(macro)'
    f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
    f1_samples.name = 'F1(samples)'
    precision_multi = PrecisionMulti(average='macro')
    recall_multi = RecallMulti(thresh=0.5, average='macro')

    # 5. Fastai datablock and dataloaders
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),                        # multi label classification
                   splitter=splitter,                                              # how we split
                   get_x=get_x,                                                    # how we acess X
                   get_y=get_y,                                                    # how we acess Y
                   item_tfms = RandomResizedCrop(IM_SIZE, min_scale=0.35))         # size & augmentation

     # create dataloaders
    dls = dblock.dataloaders(df, bs=BS)
    # show classes
    print(dls.vocab)

    # show some datas
    if SHOW_BATCH:
        dls.show_batch(nrows=2, ncols=4, figsize=(16,9))
        plt.show()

    # 6 Model
    # loss function: Flattened Binary Cross Entropy with logits Loss. Its  combine a log and sigmoid function.
    # The outputs of the model are logits

    # set up the model: resnet18, dataloaders, loss_function and metrics
    learn = vision_learner(dls, resnet18, metrics=[partial(accuracy_multi, thresh=0.5),
                                               precision_multi,
                                               recall_multi,
                                               f1_macro]).to_fp16()
    #print(learn.summary())

    # 7. Training loop
    # training loop freeze all weights of  the layers excepts the few last for 5 epochs,
    # fine tune on all the weights of the layers for 4 epochs
    if TRAIN:
        LR = 3e-3
        learn.fine_tune(3, base_lr=LR, freeze_epochs=5)
        learn.save("res18_stage1")

    if INFERENCE:
        learn.load("res18_stage1")
        learn.show_results(figsize=(16,12))
        plt.show()
