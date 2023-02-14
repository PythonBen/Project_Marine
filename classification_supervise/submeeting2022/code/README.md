### Code for supervised multilabel images classification
In a virtual environment 
To install the requires librairies:
```
pip3 install torch torchvision 
pip install -U albumentations 
pip install pandas matplotlib
```
or use:
```
pip install -r requirements.txt
```

### Running
In the main_withdataframe.py file, you can choose with the boolean variable to train (TRAIN)
, use for inference (INFERENCE) and save or not the results (WRITE_RESULT).
To run it use your editor or do:
```
python main_withdataframe.py
```
by default, for inference, we use a train mobile_net_v3 which is a good trade off between 
memory footprint and performances.

