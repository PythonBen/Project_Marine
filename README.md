# Code pour la classification et segmentation d'images pour le projet Narval

Le dossier Rami_competition_code contient les fichiers pour la classification multilabels, utilisé pour la compétition RAMI.

Le dossier apprentissage_contraste contient les fichiers pour faire de l'apprentissage auto supervisé sur un grand nombre de données. Ensuite un classifieur de type régression logistique permet de classifier les images en mode supervisé.

Le dossier classification_supervisé/submeeting_2022 contient les fichiers pour de la classification supervisée

Le dossier rapports contient les rapports de 2021 et 2022

Le dossier segmentation_nonsupervisé contient les fichiers pour segmenter des images de manière non supervisée. Il faut utiliser les fichiers dans le sous repertoire
updated_code

Le dossier segmentation_supervisé contient un notebook utilisant la librairie fastai pour effectuer la segmentation d'images sur le jeu de données labélisé Suim Dataset 
(https://irvlab.cs.umn.edu/resources/suim-dataset), il est accompagné du papier ci dessous:
[Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/pdf/2004.01241.pdf)

## acces aux datasets
les donnés sont sur un repertoire partagé data de l'ecole pour le projet Narval
Pour Windows, le montage doit se faire automatiquement avec un compte de l'école:
 (dossier //ensieta.ecole/data/labos/stic/narval)
 
Pour Linux, on peut le monter en ligne de commande comme suit :

### créer un dossier sur le PC local
```
sudo mkdir /mnt/narval
```
### montage du dossier narval
```
sudo mount -t cifs //ensieta.ecole/data/labos/stic/narval /mnt/narval -o user=votre_id_ensta (ex zerrbe pour moi)
```
## serveur de calcul curry

Le dossier command_serveur_curry contient des commandes pour transfer des fichiers entre une machine locale et le serveur curry.
Pour utiliser le serveur curry, il faut demander un compte au près du service informatique.
La rubrique [https://calcul.ensta-bretagne.fr/doku.php](https://calcul.ensta-bretagne.fr/doku.php) contient de la documentation sur le serveur curry et ginkgo.
Il contient aussi la commande pour appeler un noeud GPU. Les noeuds GPU sur le serveur curry on 32 Giga de RAM, donc c'est bien utile pour
entrainer des modèles pour le traitement d'images par exemple.

Si on utilise un editeur comme spyder par exemple, on peut le lancer avec la commande:
mesa spyder.

Pour pouvoir se connecter avec une interface graphique, le logiciel X2Go est bien pratique

