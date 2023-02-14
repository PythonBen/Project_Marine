### installation
create a virtual environnement and use
```
 pip install -r requirements.txt
```
 
### running
select the parametes in the config_param.py file and choose if you want to train the contrastive unsupervised part or the supervised logistic classier part.
The unsupervised part, will required a lots of epochs and train a large dataset.
Then you can run with the command: python main_puretorch_v3.py

If you choose to run the contrastive training part (unsupervised training part), you can set TRAIN_SU = True in the config_param.py file.
In this case, make sure that the path to save the best model weights is in your local directory, not the ensta serveur.

We can also use the library pytorch-lighning, and run the file main_addmetrics.py

To see the results for the unsupervised training with tensorboard: 
```
go to ./saved_models/Barlow_twins, then :
tensorboard --logdir=lightning_logs
```
To see the results for the supervised logistic regression with tensorboard:
```
go to ./saved_models/LogisticRegression then:
tensorboard --logdir=lightning_logs
```

### folders
in the current working directory, the folder "saved_model_purtorch" will be created, and the weighs of the model will be saved
the folders run_us and runs_su will also be created when running the unsupervised and supervised part(logistic regression).

To see the result, you can use tensorboard (pytorch environment) and go to the folder run_us for exemple and:

tensorboard --logdir=exp
then open a browser and go to the adress: http://localhost:6006/
