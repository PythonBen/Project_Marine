### installation
create a virtual environnement and use
```
pip install -r requirements.txt
```
 
### running
select the parametes in the config_param.py file and choose if you want to train the contrastive unsupervised part or the supervised logistic classier part.
The unsupervised part, will required a lots of epochs and train a large dataset.
Then you can run with the command: python main.py

With the current dataaset, it is better not to train the unsupervised part, because it takes times, and there was not real improvement 
by augmenting the batch size or the image size. According to the paper, in fact we should train on very large dataset and
large batch size, so it need large GPU RAM. 

The current saved model for the unsupervised part, which is used later with the logistic model gives satisfactory result 
on multilabel classification on our current submeeting 2022 dataset.



To see the results for the unsupervised training with tensorboard:
```
go to ./saved_models/ContrastiveLearning/SimCLR, then :
tensorboard --logdir=lightning_logs
```
To see the results for the supervised logistic regression with tensorboard:

```
go to ./saved_models/ContrastiveLearning/LogisticRegression then:
tensorboard --logdir=lightning_logs
```
