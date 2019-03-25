# Restricted-Boltzmann-Machines
Implementation of restricted Boltzmann machines in Tensorflow 2

<img src="/pictures/sampling.gif" width="320" height="320"/>





## Getting started


### Requirements
Create a virtual environment and install all required packages:

``` bash

conda create --name RBM python=3.6

source activate RBM

pip install tensorflow==2.0.0-alpha0 

pip install --upgrade tb-nightly

pip install -r requirements.txt
``` 
### Tensorboard 
In machine learning, to improve something you often need to be able to measure it.
TensorBoard is a tool for providing the measurements and visualizations needed during 
the machine learning workflow. In our case we can monitor different quantities that give important 
information about the learning process, reconstruction cross entropy, reconstruction mean squared error,
pseudo log likelihood. Moreover we can also keep track of the statistics of different parameters such as 
the weights and the biases during the learning to collect information about their behaviour during the learning. 
To use tensorboard you can use the following commands: 

``` bash

source activate RBM

tensorboard --logdir=path/to/logs
``` 
In your browser you just need to go to http://localhost:6006/. 
