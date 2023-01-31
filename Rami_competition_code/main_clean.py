from fastai.vision.all import *
import re
import gradio as gr
import cv2
from torchvision import transforms
import time

def define_paths():
    """paths where the images are located,
    path with the saved model,
    path for the dataframe"""

    path_to_class = Path('/media/ben/Data_linux/RAMI/rami_folder/home/rami/rami_marine_dataset/mastermodel/')
    path_models = Path('/media/ben/Data_linux/RAMI/code/models_multi/')
    path_root = Path('/media/ben/Data_linux/RAMI/rami_marine_dataset/')
    path_saved_dataframe  = Path('/media/ben/Data_linux/RAMI/code/dataframe_fastai/')
    #path_validation_set = Path('/media/ben/Data_linux/RAMI/rami_folder/home/rami/rami_marine_dataset/chosen_validation_data')
    path_validation_set = Path('/media/ben/Data_linux/RAMI/rami_folder/home/rami/rami_marine_dataset/test_folder')
    return path_to_class, path_models, path_root, path_saved_dataframe, path_validation_set

class BuildDataframe:
    """ class to build a dataframe with columns,
    image_to_path, Classes and labels"""

    def __init__(self, path_root=None):
        self.path_root=path_root
        self.classes = [f"class_{i}" for i in range(1,6)]
        self.list_images = []
        self.list_labels = []

    def build_dataframe(self, list_images, list_labels):
        """ method to construct a pandaDataframe from lists"""
        col_names =["image_path","labels"]
        df = pd.DataFrame(columns = col_names)
        df["image_path"] = list_images
        df["labels"] = list_labels

        return df.reset_index(drop=True)

    def make_final_dataframe(self,df):
        """ method to make the final dataframe"""

        # adding a class column
        df["Classes"] = df["image_path"].str.extract(pat=r"(class)_(\d)")[1]
        df_new = df.copy()
        df_new['lbl'] = "c" + df_new['Classes'] + "_" + df_new['labels']

        return df_new.reset_index(drop=True)

    def build_list_and_df(self):
        """ we build separately the classes in two groups.
        The first group, class 1,2 and 3 have labels indicated by its
        subfolders. The second group is class 4 and 5.
        class 1: red, white, yellow
        class 2: number1,number2,number3,number4
        class 3: number3, number4, number5, number6
        class 4: red marker on pipe
        class 5: no object of interest, (noopi)"""

        labels_classe4and5 = ['pipe','noopi']

        for i in range(3):
            cls = self.classes[i]
            labels = [item.name for item in (path_root/cls).iterdir()]
            for l in labels:
                for im in (self.path_root/cls/l).iterdir():
                    self.list_images.append(f"{cls}/{l}/{im.name}")
                    self.list_labels.append(l)

        for ix, i  in enumerate(range(3,5)):
            cls = self.classes[i]
            label = labels_classe4and5[ix]
            for im in (path_root/cls).iterdir():
                self.list_images.append(f"{cls}/{im.name}")
                self.list_labels.append(label)

        df = self.build_dataframe(self.list_images, self.list_labels)
        df = self.make_final_dataframe(df)
        print("build final dataframe")
        return df

def plotting():
    """"plottitng the distribution of classes and labels"""
    df.groupby("Classes").count().plot(kind='pie', y="labels", figsize=(10,10))
    df.groupby("lbl").count().plot(kind='pie', y="labels", figsize=(10,10))
    plt.show()

def saving_exporting_model(learn=None, name="learn18_multi_stage1"):
    """ function to save and export a train model"""
    learn.save(path_models/name)
    learn.export(path_models/name)

# testing
def make_list_for_testing(path=None):
    """
    :param path: path to folder which contain image to test
    :return: list of path
    """
    list_test_images = []
    for item in path.iterdir():
        #print(item.name)
        list_test_images.append(item.name)
    return list_test_images

# making predictions
def test_valid():
    list_test_images = make_list_for_testing(path=path_validation_set)
    for i in range(len(list_test_images)):
        predict_class = learn18_inf.predict(path_validation_set/list_test_images[i])
        p = predict_class[0]
        print(p)

# using gradio app to deploy the model

def predict_func(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn18_inf.predict(img)
    #return {labels[i]: float(probs[i]) for i in range(len(labels))}
    return pred_idx, probs

#pred_idx, probs = predict_func(path_validation_set/list_test_images[0])
#print("pred_idx")
#print(pred_idx.numpy())
#print("probs")
#print(probs)


#  using the Class Activation Map, we will find the centroid
finalconv_name = 7
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def prepare_cam(learner=None):
    net = (learner.model).eval().cpu()

    net._modules.get("0")[finalconv_name].register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-1].data.numpy())
    #print(weight_softmax.shape)
    return net, params, weight_softmax

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 128x128
    size_upsample = (128,128)
    bz, nc, h, w = feature_conv.shape
    #print(f"b:{bz}, nc:{nc}, h:{h}, w:{w}")
    output_cam = []
    for idx in class_idx:
        beforeDot = feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight_softmax[idx],beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def making_marker(path=None, list_images=None, learner=None, weight_softmax=None, N=None,classes=None):

    # load test image
    image_file = path/list_images[N]
    #print(image_file)
    img_variable = PILImage.create(image_file)
    #print(f"img_variable:{img_variable.shape}")
    pred,pred_idx,probs = learner.predict(img_variable)      # fastai resize the input images automatically like the image used in the dataloader for training
    idx = [pred_idx.numpy()]
    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    img = cv2.imread(image_file.as_posix())
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.1 + img * 0.9
    result_01 = result/255
    if classes[idx[0]] !="c5_noopi":
        result_indices = np.where(heatmap == np.amax(heatmap))
        x_coord = int(result_indices[1].mean())
        y_coord = int(result_indices[0].mean())

    else:
        x_coord, y_coord = None, None
    return (x_coord, y_coord), result_01, img, pred, pred_idx, img_variable.shape

def plotting_sample(path=None, list_images=None, learner=None, weight_softmax=None, N=None, classes=None):
    coord , result_01, img, pred, pred_idx, _ = making_marker(path, list_images, learner, weight_softmax, N, classes)
    print(coord)
    plt.imshow(result_01)
    print("prediction")
    print(pred)
    if (coord[0] is not None) and (coord[1]) is not None:
        plt.scatter(coord[0], coord[1], s=50, c='red', marker='o')
        plt.show()
    plt.show()

def write_outputfile(path_output_file=None, list_test_images=None, learner=None, path_validation_set=None, weight_softmax=None, classes=None):

    #predictions_file = "/media/ben/Data_linux/RAMI/code/predictions/predicts3.txt"
    prediction_file = path_output_file/"output.txt"
    prediction_file.touch(exist_ok=True)
    pattern = r"([a-z])(\d)_(\w+)"
    with open(prediction_file, 'w') as f:
        time_file = int(time.time())
        for i in range(len(list_test_images)):
            coord , _, _,_,_,_ = making_marker(path_validation_set, list_test_images, learner, weight_softmax,i, classes) # to correct
            pred = learner.predict(path_validation_set/list_test_images[i])[0]
            findpattern = re.search(pattern, pred)
            if findpattern is not None:
                if findpattern.group(2) in ['4', '5']:             #  if classe 4 or 5 no instance.
                    pat = ''                             # no instance for classes 4 and 5
                else:
                    if findpattern.group(3)[0:6] == "number":      # get numbers for classes 2 and 3
                        pat = findpattern.group(3)[7]    # gey the number
                    else:
                        pat = findpattern.group(3)       # get colors for classes 1
                if coord[0]==None:
                    f.write(f"{time_file},{findpattern.group(2)},{pat},{''},{''} \n")
                else:
                    f.write(f"{time_file},{findpattern.group(2)},{pat},{coord[0]},{coord[1]} \n")

# to do use fastai to observe any activation inside the model
# CAM and HOOKS, class activation map. The idea is to plot and find the maximum action for different layer

class Hook():
    """ hook function to store a copy of the output"""
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class HookBwd():

    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()



if __name__ == "__main__":
    # parameters
    GR = False # deploy web app with gradio
    Nimage = 0
    PLOT_CLASS = False
    WRITE_OUTPUT = False
    SIZE = (128*2,128*2)
    BS = 6
    PRINT_DF = True
    PLOT_BATCH_INF = True
    # paths
    path_to_class, path_models, path_root, path_saved_dataframe, path_validation_set = define_paths()
    # dataframe
    if (path_saved_dataframe/"df.pkl").exists():
        print("Dataframe already built and saved as df.pkl")
        df = pd.read_pickle(path_saved_dataframe/"df.pkl")
    else:
        B1 = BuildDataframe(path_root=path_root)
        df = B1.build_list_and_df()
        #df.to_pickle(path_saved_dataframe/"df.pkl")                         # to save the dataframe, uncomment
    if PRINT_DF:
        print(df.shape)
        print(df.head())
        print(df.tail())
    if PLOT_CLASS:
        plotting()

    # create a databunch, fastai librairy, necessary to use the model later for inference
    data = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=RandomSplitter(),
                       get_x=ColReader('image_path', pref=str(path_root)+os.path.sep),
                       get_y=ColReader('lbl'),
                       item_tfms = Resize(SIZE),)

    dls = data.dataloaders(df, bs = BS)
    classes = dls.vocab
    print(f"classes:{classes}")

    # model
    # load  the model for inference of train if necessary
    # define model and metrics
    f1_score = F1Score(average='macro')
    precision = Precision(average='macro')
    recall = Recall(average='macro')

    learner = vision_learner(dls, resnet18, metrics=[accuracy, f1_score,precision,recall])

    if (path_models/"learn18_multi_stage1.pkl").exists():
    #if (path_models/"learn18_multi_stage_essai.pkl").exists():
        #learn18_inf = learn18.load(path_models/"learn18_multi_stage1")
        learn18_inf = learner.load(path_models/"learn18_stage1_im256x256_bs6")
    else:
        LR = 1e-3
        learner.fine_tune(3, base_lr=LR, freeze_epochs=3)
        learn18_inf = learner

    if PLOT_BATCH_INF:
        learn18_inf.show_results()
        plt.show()
    if GR:
        labels = learn18_inf.dls.vocab
        gr.Interface(fn=predict_func, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3)).launch(share=True)


    #  using the Class Activation Map, we will find the centroid
    list_images = make_list_for_testing(path=path_validation_set)
    finalconv_name = 7
    features_blobs = []

    net, params, weight_softmax = prepare_cam(learner=learn18_inf)
    (x_coord, y_coord), result_01, img, pred, pred_idx, img_orig_shape = making_marker(path=path_validation_set,
                                                                                       list_images=list_images,
                                                                                       learner=learn18_inf,
                                                                                       weight_softmax=weight_softmax,
                                                                                       N=Nimage,
                                                                                       classes=classes)

    plotting_sample(path=path_validation_set,
                    list_images=list_images,
                    learner=learn18_inf,
                    weight_softmax=weight_softmax,
                    N=Nimage,
                    classes=classes)

    if WRITE_OUTPUT:
        path_output_file = Path('/media/ben/Data_linux/RAMI/code/predictions')
        write_outputfile(path_output_file, list_images, learn18_inf, path_validation_set, weight_softmax,classes)

    # heatmap of different layers

    img = PILImage.create(path_validation_set/list_images[Nimage])
    x, = first(dls.test_dl([img]))
    print(f"x size")
    print(x.size())
    x_dec = TensorImage(dls.train.decode((x,))[0][0])
    cls = pred_idx
    print(f"cls:{cls}")
    Nlayer = -1
    with HookBwd(learn18_inf.model[0][Nlayer]) as hookg:
        with Hook(learn18_inf.model[0][Nlayer]) as hook:
            output = learn18_inf.model.eval()(x.cuda())
            act = hook.stored
        output[0,cls].backward()
        grad = hookg.stored

    w = grad[0].mean(dim=[1,2], keepdim=True)
    cam_map = (w * act[0]).sum(0)
    print("cam_map")
    print(cam_map.size())
    map_size = cam_map.shape[0]
    result_max = ((cam_map == cam_map.max()).nonzero()).squeeze().cpu()
    print("result_max")
    print(result_max)                # to do convert into h and w, try different layers
    h_orig, w_orig = img_orig_shape

    xcor = result_max[1]*w_orig/map_size
    ycor = result_max[0]*h_orig/map_size


    print(f"xcor:{xcor}, ycor:{ycor}")

    _,ax = plt.subplots()
    x_dec.show(ctx=ax)
    ax.imshow(cam_map.detach().cpu(), alpha=0.5, extent=(0,128*2,128*2,0),interpolation='bilinear', cmap='magma')
    #plt.scatter(ycor, xcor, s=50, c='red', marker='o')
    plt.show()

    #plt.imshow(cam_map.detach().cpu())
    plt.imshow(result_01)
    plt.scatter(xcor,ycor, s=50, c='red', marker='o')

    plt.show()



