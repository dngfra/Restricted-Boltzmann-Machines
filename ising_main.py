import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from my_RBM_tf2 import RBM
import deepdish as dd
import itertools
import os

datah5 = dd.io.load('data/ising/ising_data_complete.hdf5')

#Transform -1 in 0 and take spin up as standard configuration
binarizer = Binarizer(threshold=0)
keys = list(datah5.keys())
for key in keys:
    datah5[key] = np.array([np.where(np.sum(slice)>0,-slice,slice) for slice in datah5[key]])
    datah5[key] = np.array([binarizer.fit_transform(slice) for slice in datah5[key]])

class_labels = np.asarray(list(itertools.chain.from_iterable(itertools.repeat(x, 5000) for x in range(0,6))))
class_names = ['T=1.000000', 'T=2.186995', 'T=2.261435', 'T=2.268900', 'T=2.269184', 'T=3.000000'] # equivalent to keys

data = datah5['T=1.000000']
for temperature in class_names[1:]:
    data = np.concatenate([data,datah5[temperature]])

#create dictionary for training
x_train, x_test, y_train, y_test = train_test_split(data, class_labels, test_size=0.2, random_state=42)


#reshape pictures to be vectors and fix datatype
x_train = x_train.reshape(x_train.shape[0],-1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0],-1).astype(np.float32)

data = {"x_train": x_train ,"y_train": y_train,"x_test": x_test,"y_test": y_test}

#Create a restricted boltzmann machines
machine = RBM(x_train[0].shape[0], 200, 1000,(32,32), 32, optimizer='pc')

#Train the machine
machine.train(data)
