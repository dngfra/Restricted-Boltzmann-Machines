from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from my_RBM_tf2 import RBM
import deepdish as dd
import itertools
import os
from optimizer import Optimizer


datah5 = dd.io.load('data/ising/ising_data_complete.hdf5')

#Transform -1 in 0 and take spin up as standard configuration
binarizer = Binarizer(threshold=0)
keys = list(datah5.keys())
#put here the temperature from keys that you want to use for the training
class_names = ['T=2.186995', 'T=2.269184', 'T=3.000000']
datah5_norm={}
data_bin={}
for key in keys:
    datah5_norm[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
    data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]])

#class labels even if they are not really useful here
class_labels = np.asarray(list(itertools.chain.from_iterable(itertools.repeat(x, 5000) for x in range(0,len(class_names)))))

data = data_bin[class_names[0]]
for temperature in class_names[1:]:
    data = np.concatenate([data,data_bin[temperature]])

#create dictionary for training
x_train, x_test, y_train, y_test = train_test_split(data, class_labels, test_size=0.1, random_state=42)

#reshape pictures to be vectors and fix datatype
x_train = x_train.reshape(x_train.shape[0],-1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0],-1).astype(np.float32)

data = {"x_train": x_train ,"y_train": y_train,"x_test": x_test,"y_test": y_test}

#Create a restricted boltzmann machines
machine = RBM(x_train[0].shape[0], 1200, 100, (32, 32), 128,'cd')

optimus = Optimizer(machine, 0.1, opt = 'adam')
#Train the machine
machine.train(data,optimus)