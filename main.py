import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Binarizer
#import matplotlib.pyplot as plt
from my_RBM_tf2 import RBM

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.average(x_train/255)

x_train = x_train/255
x_test = x_test/255


binarizer = Binarizer(threshold=0.5)
x_train_binary = np.array([binarizer.fit_transform(slice) for slice in x_train])
x_test_binary = np.array([binarizer.fit_transform(slice) for slice in x_test])

#plt.matshow(x_train_binary[6].reshape(28,28))
#plt.show()

x_train_binary = x_train_binary.reshape(x_train_binary.shape[0],-1).astype(np.float32) #[:1600,:]
x_test_binary = x_test_binary.reshape(x_test_binary.shape[0],-1).astype(np.float32) #[:1600,:]

data = {"x_train": x_train_binary ,"y_train": y_train,"x_test": x_test_binary,"y_test": y_test}

machine = RBM(x_train_binary[0].shape[0], 200, 100, 32)

machine.train(data)

