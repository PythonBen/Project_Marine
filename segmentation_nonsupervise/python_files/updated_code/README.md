l# Unsupervised_segmentation_diffclustering
This code is based on the paper: 
[Unsupervised learnng of image segmentation based on differentiable feature clustering (W, Kim and A, Kanezaki and M, Tanaka)](https://arxiv.org/abs/2007.09990)

The github repo is:https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip

## Getting started
To segment one image, you can use the programme main_one_image.py.

You must provided the input_path where is locatated the input image and the result path.


For exemple:
```

python main_one_image --path_input /user/my_path --path_output /user/my_result --input_filename image.jpg --output_filename image_result.jpg

```
The programme main.py is used to segment multiples images after training on multiples images.
```
python main.py --path_root --train_folder --valid_folder --result_folder
```
The train, valid and result folder should be located in the path_root_folder

## Installation

Is better to create a virtual environment and use pip:

You need to intall torch, numpy, opencv-contrib-python and matplotlib librairies with the command:
```
pip install numpy opencv-contrib-python torch matplotlib
```
or

you can create a virtual environnement and install all the package with the requirements.txt file with the command
```
pip install -r requirements.txt
```

## Main parameters that influence the segmentation:
1. nChannel:  The dimension of the pixel vector in the feature space (between 40 and 100)
2. lr: learning rate (0.001 to 0.01)
3. min_labels: number of final labels we want (4 or 5 for example)
4. stp_con: weight for the continuity loss function
5. stp_sim: weight for the similarity loss function
   

