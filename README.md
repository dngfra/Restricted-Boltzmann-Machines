# Restricted-Boltzmann-Machines and VAE
Implementation of restricted Boltzmann machines and Variational Autoencoder in Tensorflow 2

<img src="/pictures/sampling_speed.gif" width="210" height="210"/>  <img src="/pictures/rbm2.png" height="190"/>

## What is implemented 
- Bernoulli RBM 
- Contrastive Divergence, Persistent Contrastive Divergence 
- KL-Divergence via neighbours distance measure 
- Exact partition function Z for small models 
- Approximate partition function  Z via Annealed Importance Sampling 
- Log likelihood using AIS 
- Pseudo Log likelihood 
- Autoencoder (with conv layers)
- Variational autoencoder VAE 
- Conditional variational autoencoder c-VAE
- Ising model: Energy, Two points correlation, correlation lenght
- Tensorboard: Variable histograms, Reconstruction cross entropy, mean squared error, KL divergence, inverse KL divergence, log-likelihood, gradients, 
visualization of samples, text summary for the model parameters.



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
## Basic Usage 
### Training an RBM
The first step to train our Restricted Boltzmann machine is to create it. At the moment we can only crate binary or Bernoulli RBM. After we imported the required classes we can initialize our machine calling RBM and specifying the following parameters: RBM(visible units, hidden units, number of epochs, input picture shape, batch size, optimization algorithm('cd' or 'pcd'), inizialization weights, number of MC steps, l1).
Together with the machine we also need an optimizer that has to be initialized with an RBM object, the initial learning rate, and the optimization algorithm ('adam' or 'SGD'). The last thing that we need to inizialize is a metrics_monitor, it is a class that collect some of the metrics useful to monitor the learning and the performance of the machine. 
``` python
from RBM import RBM
from optimizer import Optimizer
from performance_metrics import Metrics_monitor
from utils import plot_image_grid, plot_single_image, plot_input_sample

machine = RBM(784, 200,100,(28,28), 128, 'cd', initializer = 'normal')
optimus = Optimizer(machine, 0.1, opt = 'adam')
machine.save_param(optimus)
monitor = Metrics_monitor(machine)
#Train the machine
machine.train(data,optimus,monitor)
``` 
Given that we are dealing with Bernoulli RBM the input data must be binarized (0,1) (see main.py for more details). 
With the preprocessed data we can create a dictionary that will be used to train the machine. 
``` python
data = {"x_train": x_train_binary ,"y_train": y_train,"x_test": x_test_binary,"y_test": y_test}
``` 
Train the machine:
``` python
machine.train(data,optimus)
``` 
The model parameters are automatically saved in .h5 file every epoch. 
### Sample from an RBM 
Given some trained parameters, we want to rebuild our model from the saved configuration and sample new datapoints from the data distribution that we learnt, this follows straightforward. First of all, we have to rebuild our model from the saved configuration using *.from_saved_model(path)*. 
As we know, to sample a new point we have to perform alternating Gibbs sampling between the visible and hidden layers, using *.sample* we can do this 
starting the Markov chain from a real datapoint (if we specify *inpt*) or from random noise for which we can specify the distribution  of zeros and ones (default 0.5). 
``` python
machine = RBM(784, 200,100,(28,28), 128, 'cd')
machine.from_saved_model(path)

visible_states_1,visible_probabilities_1,inpt,evolution_MC= machine.sample(n_step_MC=5000)
plot_input_sample(inpt,fantasy_particle1,(28,28))
``` 
In the code we also used the function  *plot_input_sample()* from *utils* to plot the input and the sample. The method *.sample()* outputs other objects that could be useful for some analysis like a list containing the entire set of visible state steps of the markov chain.
We use the latter to generate the gif at the beginning of the page. 

<img src="/pictures/sample.png" height="240"/> 

### Inspect the weights 
Given a trained machine it could be useful to visually inspect the weights or the features of the data that the machine is learning. To do so we can plot 
the weights of each hidden units reshaped as the input pictures so that we can understand and see what, or which part of the picture is "activating" the hidden neurones. This
is sometimes also called the receptive field for an analogy with what is happening with ganglion cells, rods and cones in the biological retina. 
To do this we can use *plot_image_grid* from *utils* giving the weights of the machine. 

``` python
#using the same machine that we rebuild before
image_shape = (28, 28) # 28x28 = 784 pixels in every image
weights = np.asarray(machine.weights) 

plot_image_grid(weights, image_shape,9, save = True)


``` 

<img src="/pictures/weights.png" height="320"/>

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
