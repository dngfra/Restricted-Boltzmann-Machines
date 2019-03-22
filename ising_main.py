import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from my_RBM_tf2 import RBM
import deepdish as dd

data = dd.io.load('data/ising/ising_data.hdf5')

data_T2 = data['T=2.000000']
data_T2_26 = data['T=2.269000']
data_T2_5 = data['T=2.500000']

binarizer = Binarizer(threshold=0)
data_T2_binary = np.array([binarizer.fit_transform(slice) for slice in data_T2])
data_T2_26_binary = np.array([binarizer.fit_transform(slice) for slice in data_T2_26])
data_T2_5_binary = np.array([binarizer.fit_transform(slice) for slice in data_T2_5])
label_T2 = np.ones((1000))*2



x_train, x_test, y_train, y_test = train_test_split(data_T2_binary, label_T2, test_size=0.1, random_state=42)


#reshape pictures to be vectors and fix datatype
x_train = x_train.reshape(x_train.shape[0],-1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0],-1).astype(np.float32)

data = {"x_train": x_train ,"y_train": y_train,"x_test": x_test,"y_test": y_test}

#Create a restricted boltzmann machines
machine = RBM(x_train[0].shape[0], 200, 1000, 32)

#Train the machine
machine.train(data)
