"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import Binarizer
import numpy as np
import tensorflow as tf
import os
import deepdish as dd
import itertools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

original_dim = 32*32


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_layer_0 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.hidden_layer_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.hidden_layer_5 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        #self.output_layer = tf.keras.layers.Dense(10, activation=tf.keras.activations.linear) #tf.keras.activations.linear activation=tf.nn.softmax
        self.output_layer = tf.keras.layers.Dense(9, activation=tf.nn.relu)
    def call(self, input_features):
        activation = self.hidden_layer_5(self.hidden_layer_4(self.hidden_layer_3(self.hidden_layer_2(self.hidden_layer_1(self.hidden_layer_0(input_features))))))
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, original_dim):
        super(Decoder, self).__init__()
        #self.hidden_layer_1 = tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)
        self.hidden_layer_1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.hidden_layer_5  = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.hidden_layer_6 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.sigmoid)

    def call(self, code):
        activation = self.hidden_layer_6(self.hidden_layer_5(self.hidden_layer_4(self.hidden_layer_3(self.hidden_layer_2(self.hidden_layer_1(code))))))
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    def __init__(self, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


if __name__ == '__main__':
    '''
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
    x_train_binary = x_train_binary.reshape(x_train_binary.shape[0],-1).astype(np.float32)
    x_test_binary = x_test_binary.reshape(x_test_binary.shape[0],-1).astype(np.float32)
    '''

    here = os.path.dirname(os.path.abspath(__file__))

    datah5 = dd.io.load(here + '/data/ising/ising_data_L32.hdf5')

    # Transform -1 in 0 and take spin up as standard configuration
    binarizer = Binarizer(threshold=0)
    keys = list(datah5.keys())
    # put here the temperature from keys that you want to use for the training
    class_names = [keys[i] for i in [4, 6, 7, 8, 9, 10, 11, 12, 16]]
    #class_names = [keys[9]]
    n_samples = datah5[keys[0]].shape[0]
    datah5_norm = {}
    data_bin = {}
    for key in keys:
        datah5_norm[key] = np.array([np.where(np.sum(slice) < 0, -slice, slice) for slice in datah5[key]])
        data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]])

    # class labels even if they are not really useful here
    class_labels = np.asarray(
        list(itertools.chain.from_iterable(itertools.repeat(x, n_samples) for x in range(0, len(class_names)))))
    one_hot_labels = np.zeros((len(class_labels), len(class_names)))
    one_hot_labels[np.arange(len(class_labels)), class_labels] = 1

    data = data_bin[class_names[0]]
    for temperature in class_names[1:]:
        data = np.concatenate([data, data_bin[temperature]])

    # create dictionary for training
    x_train, x_test, y_train, y_test = train_test_split(data, class_labels, test_size=0.1, random_state=42)

    # reshape pictures to be vectors and fix datatype
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    checkpoint_path = "training_autoencoder/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=1)

    autoencoder = Autoencoder(original_dim=original_dim)
    autoencoder.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #autoencoder.summary()

    autoencoder.fit(x_train, x_train , epochs=100,batch_size=64,validation_data = (x_test,x_test), callbacks = [cp_callback])
    autoencoder.save_weights('auto_weights_tanh.h5')

    test = np.zeros((50, 32*32))
    for i in range(9):
        mask = np.where(y_train == i)
        ciao = x_train[mask]
        ciao2 = ciao[:5]
        test[i * 5:i * 5 + 5] = ciao2

    indx = np.random.randint(0, x_test.shape[0], 20)
    fig, axes = plt.subplots(nrows=50, ncols=3, figsize=(75, 75))
    for en in range(50):
        reconstruction_plot = np.random.binomial(1, autoencoder(test[en:en + 1, :]))
        encoded = autoencoder.encoder(test[en:en + 1, :])
        print(np.argmax(encoded.numpy()))
        x = np.arange(encoded.shape[1])
        axes[en][0].imshow(test[en, :].reshape(32, 32), cmap='Greys')
        axes[en][0].set_title("Original Image")
        axes[en][1].imshow(np.asarray(reconstruction_plot).reshape(32, 32), cmap='Greys')
        axes[en][1].set_title("Reconstruction")
        axes[en][2].bar(x, encoded.numpy().reshape((9)))
    plt.savefig('plot_test.pdf')


