we have participate to the RAMI(Robotics for Asset Maintenance and Inspection) competition in the framework of the METRICs EU project.
We have use a classic Resnet18 neural network and prepare the dataset for multilabel classification.
The results are [here](https://metricsproject.eu/inspection-maintenance/rami-cascade-campaign-results-marine/)

The rule book  and the details competition is here:
https://metricsproject.eu/inspection-maintenance/rami-cascade-campaign-results-marine/

A docker image is available here:https://hub.docker.com/repository/docker/blepers/rami_v2/general

## Installation

The librairies used are:
- torch
- fastai
- opencv (cv2)
- gradio (for webapp)

Create a virtual environment and install them with the command:
```
pip install torch torchvision fastai open-cv-contrib-python gradio
```
you can also install all with a virtual environment an use the command:
```
pip install -r requirements_fastai.txt
```
## running
in a terminal use ./run.sh

you can also use the file main_clean.py with different options, training or inference
