# Ad-hoc Federated Learning For Edge-Devices
Faraz Barati (faraz.barati@utexas.edu), Vignesh Nandakumar (vnandakumar@utexas.edu)

DISCLAIMER: This is a forked repository. The original repo can be found here: [Federated Learning Pytorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch)

This repo provides the code necessary to create simulations of mobile nodes which aggregate their neural-networks via Federated Learning (FL) to create a global model which can classify images. 
Experiments are produced on MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data is distributed via the Dirichlet distribution, where alpha = 0.1.

The main file, ```federated_adhoc.py```, determines which nodes are available in a communication round and how they communicate with one another. There are three simulation methods:
* Centralized: all nodes in a communication round are connected to the cloud.
* Random: clusters are initialized randomly, where a selected node can communicate with the cloud and its neighbors' features are aggregated.
* Modularity: clusters are initialized through a modularity algorithm. Then, the node with the highest degree is deemed the selected node. Feature aggregation occurs from the selected nodes' neighbors.

In a communication round, connectivity issues can be included by toggling the probability of a round facing issues and the number of nodes removed. More details can be found below.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on the MNIST and CIFAR10.
* Mobility data is acquired from the [Foursquare dataset](https://drive.google.com/file/d/1sFOMHPZOCiVKCVQAVubxyEApXWu-eU81/view). Our [data](data/Top_250_Clients_9_to_5.csv) is a modified version of the original dataset, where only the top 250 nodes most occuring nodes from 9 AM - 5 PM are selected.

## Running the experiments
In the root directory, run the following commands:
```
cd src
python federated_adhoc.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python federated_adhoc.py --model=mlp --dataset=mnist --gpu=1 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 250.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--central:```  Running experiments where every node is accounted for (STAR Topology). Either 0 or 1.
* ```--modularity:```  Running experiments where modularity algorithm clusters nodes. Either 0 or 1.
* ```--p_round_fail:``` Sets the probability of a round facing connectivity issues. Range is [0,1].
* ```--num_nodes_rem:``` Number of nodes removed in a communication round. Default is 0. Range is [0,1].

## Plotting
If ```src/experiments.sh``` was ran, then running ```python plot_experiments.py``` in the ```src``` directory will generate bar graphs for the simulations in the bash script. An example can be found [here](save/prl_sweep.png).

Otherwise, ```src/plot.py``` is fit if the following commands were ran.
```
python federated_adhoc.py --model=cnn --dataset=mnist --epochs=250 --modularity=0 --central=0 --verbose=0 --iid=0
python federated_adhoc.py --model=cnn --dataset=mnist --epochs=250 --modularity=0 --central=1 --verbose=0 --iid=0
python federated_adhoc.py --model=cnn --dataset=mnist --epochs=250 --modularity=1 --central=0 --verbose=0 --iid=0
```
In this scenario, running ```plot.py``` in the ```src``` directory will generate plots for communication time per commmunication round, average cumulative battery used, and training loss across three methods. If other simulations were ran, please modify ```plot.py``` to update your parameters.