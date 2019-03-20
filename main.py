import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Binarizer
#import matplotlib.pyplot as plt
from my_RBM_tf2 import RBM

#Import mnist dataset
mnist = tf.keras.datasets.mnist

#Split in test and train
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Scale entries between(0,1)
x_train = x_train/255
x_test = x_test/255

#Binarize pictures
binarizer = Binarizer(threshold=0.5)
x_train_binary = np.array([binarizer.fit_transform(slice) for slice in x_train])
x_test_binary = np.array([binarizer.fit_transform(slice) for slice in x_test])

#reshape pictures to be vectors and fix datatype
x_train_binary = x_train_binary.reshape(x_train_binary.shape[0],-1).astype(np.float32)[:1600,:]
x_test_binary = x_test_binary.reshape(x_test_binary.shape[0],-1).astype(np.float32)[:1600,:]

#shuffle data
np.random.shuffle(x_train_binary)
np.random.shuffle(x_test_binary)

#create dictionary of data
data = {"x_train": x_train_binary ,"y_train": y_train,"x_test": x_test_binary,"y_test": y_test}

#Create a restricted boltzmann machines
machine = RBM(x_train_binary[0].shape[0], 200, 100, 32)

#Train the machine
machine.train(data)

