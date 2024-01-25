But : 

Installer /3b1b/ en entier.

et executer la commande "python3 tester_model.py"

Sur linux ca marche parfaitement.

Sur Windows non, faut compiler tout, et installer des trucs nvidia/cuda et c'est 100x top long.

Donc il faut emuler Linux sur Windows a travers Virtual box :


Etapes :

Installer virtualbox

Installer un .iso de Ubuntu par exemple

Installer le system Ubuntu sur Virtualbox (suivre un tuto aleatoire) (peut prendre longtemps, meme peut etre plus de 1h)

Telecharger /3b1b/ sur le nouveau system

executer les commandes

sudo apt-get install python3-pip

pip3  install matplotlib

executer python3 tester_model.py
