# Restricted-Boltzmann-Machines
Implementation of restricted Boltzmann machines in Tensorflow 2

<img src="/pictures/rbm.png" height="240"/>





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
The first step to train our Restricted Boltzmann machine is to create it. At the moment we can only crate binary or bernoulli RBM. After we imported the required classes we can initialize our machine calling RBM and specifying the following parameters: RBM(visible units, hidden units, number of epochs, input picture shape, batch size, optimization algorithm('cd' or 'pcd')).
Together with the machine we also need an optimizer that has to be initialized with a RBM object, the initial learning rate, and the optimization algorithm ('adam' or 'SGD').
``` python
from my_RBM_tf2 import RBM
from optimizer import Optimizer
from utils import plot_image_grid, plot_single_image, plot_input_sample

machine = RBM(784, 200, 100,(28,28), 32, 'cd')
optimus = Optimizer(machine, 0.1, opt = 'adam')
``` 
Given that we are dealing with Bernoulli RBM the input data must be binarized (0,1) (see main.py for more details). 
e the preprocessed data we can create a dictionary that will be used to train the machine. 
``` python
data = {"x_train": x_train_binary ,"y_train": y_train,"x_test": x_test_binary,"y_test": y_test}
``` 
Train the machine:
``` python
machine.train(data,optimus)
``` 
The model parameters are automatically saved in .h5 file every epoch. 
### Sample from an RBM 
Given some trained parameters, we want to re-build our model from the saved configuration and sample new datapoints from the data distribution that we learnt, this follow straightforward
 using First of all we have to re-build our model from the saved configuration using *.from_saved_model(path)* and *.sample()*. 
As we know, to sample a new point we have to perform alternating Gibbs sampling between the visible and hidden layers, using.sample we can do this 
starting from a real datapoint (if we specify *inpt*) or from random noise for which we can specify the distribution  of zeros and ones (default 0.5). 
``` python
machine = RBM(784, 200, 100, (28,28), 32)
machine.from_saved_model(path)

visible_states_1,visible_probabilities_1,inpt,evolution_MC= machine.sample(n_step_MC=50000)
plot_input_sample(inpt,fantasy_particle1,(28,28))
``` 
where we also used *plot_input_sample()* to plot the input and the sample. The method *.sample()* outputs other objects that could be useful for some analysis. 

<img src="/pictures/sample.png" height="240"/> <img src="/pictures/sampling.gif" width="240" height="240"/> 

### Inspect the weights 
Given a trained machine it could be usefull to visually inspect the weights or the features of the data that the machine is learning. To do so we can plot 
the weights of each hidden units reshaped as the input pictures so that we can understand and see what, or which part of the picture is "activating" the hidden neurones. This
is sometimes also called the receptive field for an analogy with what is happening with ganglion cells, rods and cones in the biological retina. 
To do this we can use *plot_image_grid* from utils giving the weights of the machine. 


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
