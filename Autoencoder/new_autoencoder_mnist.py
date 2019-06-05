"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import Binarizer
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import deepdish as dd
import itertools
from sklearn.model_selection import train_test_split


original_dim = 784


class Encoder(tf.keras.layers.Layer):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.hidden_layer_0 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.hidden_layer_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.hidden_layer_4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.hidden_layer_5 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(latent_dim, activation=tf.nn.softmax) #tf.keras.activations.linear

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
    def __init__(self, original_dim,latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed,code

def loss(model,input,labels_one_hot,alpha=1,beta=0):
    reconstruction,encoded = model(input)
    l_1 = tf.losses.binary_crossentropy(input,reconstruction,from_logits=False)
    l_2 = tf.losses.categorical_crossentropy(labels_one_hot,encoded,from_logits=False,label_smoothing=0)
    return alpha*l_1+beta*l_2

def grad(model, inputs, labels_one_hot):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, labels_one_hot)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
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

    one_hot_labels = np.zeros((60000,10))
    one_hot_labels[np.arange(60000), y_train] = 1

    dataset_train = tf.data.Dataset.from_tensor_slices(x_train_binary)
    dataset_train_labels = tf.data.Dataset.from_tensor_slices(y_train)
    dcombined = tf.data.Dataset.zip((dataset_train, dataset_train_labels)).batch(32)


    data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    checkpoint_path = "training_autoencoder/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=1)

    autoencoder = Autoencoder(original_dim=original_dim, latent_dim=10)
    #autoencoder.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    ## Note: Rerunning this cell uses the same model variables

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        reconstruction_accuracy = tf.keras.metrics.Accuracy()
        # Training loop - using batches of 32
        for x, y in dcombined:
            # Optimize the model
            loss_value, grads = grad(autoencoder, x, y)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(np.argmax(y,1), autoencoder(x)[1])
            reconstruction_accuracy(x,np.random.binomial(1,autoencoder(x)[0]))
        # end epoch
        #train_loss_results.append(epoch_loss_avg.result())
        #train_accuracy_results.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Reconstruction: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result(),
                                                                        reconstruction_accuracy.result()))
        reconstruction_plot = autoencoder(inpt=x_test_binary[1, :])[0]
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(x_test_binary[1, :].reshape(28,28), cmap='Greys')
        axes[0].set_title("Original Image")
        axes[1].imshow(np.asarray(reconstruction_plot).reshape(28,28), cmap='Greys')
        axes[1].set_title("Reconstruction")
        plt.show(block=False)
        plt.pause(3)
        plt.close()