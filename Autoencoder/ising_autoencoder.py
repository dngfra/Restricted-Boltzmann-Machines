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


original_dim = 1024


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

    datah5 = dd.io.load(here + '/data/ising/ising_data_L32.hdf5')

    # Transform -1 in 0 and take spin up as standard configuration
    binarizer = Binarizer(threshold=0)
    keys = list(datah5.keys())
    # put here the temperature from keys that you want to use for the training
    class_names = [keys[i] for i in [4, 6, 7, 8, 9, 10, 11, 12, 16]]
    n_samples = datah5[keys[0]].shape[0]
    datah5_norm = {}
    data_bin = {}
    for key in keys:
        datah5_norm[key] = np.array([np.where(np.sum(slice) < 0, -slice, slice) for slice in datah5[key]])
        data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]])

    # class labels even if they are not really useful here
    class_labels = np.asarray(
        list(itertools.chain.from_iterable(itertools.repeat(x, n_samples) for x in range(0, len(class_names)))))
    one_hot_labels = np.zeros((len(class_labels),len(class_names)))
    one_hot_labels[np.arange(len(class_labels)), class_labels] = 1

    data = data_bin[class_names[0]]
    for temperature in class_names[1:]:
        data = np.concatenate([data, data_bin[temperature]])

    # create dictionary for training
    x_train, x_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.1, random_state=42)

    # reshape pictures to be vectors and fix datatype
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    dataset_train = tf.data.Dataset.from_tensor_slices(x_train)
    dataset_train_labels = tf.data.Dataset.from_tensor_slices(y_train)
    dcombined = tf.data.Dataset.zip((dataset_train, dataset_train_labels)).batch(32)


    data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    checkpoint_path = "training_autoencoder/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=1)

    autoencoder = Autoencoder(original_dim=original_dim, latent_dim=len(class_names))
    autoencoder.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

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

        ind = np.random.randint(0,x_test.shape[0],1)[0]
        print(ind)
        reconstruction_plot = np.random.binomial(1,autoencoder(x_test[ind:ind+1, :])[0])
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(x_test[ind, :].reshape(32,32), cmap='Greys')
        axes[0].set_title("Original Image")
        axes[1].imshow(np.asarray(reconstruction_plot).reshape(32,32), cmap='Greys')
        axes[1].set_title("Reconstruction")
        plt.show(block=False)
        plt.pause(3)
        plt.close()