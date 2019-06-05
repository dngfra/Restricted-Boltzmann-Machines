from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import datetime
import imageio
import deepdish as dd
import itertools
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from IPython import display
import tensorflow as tf

#os.chdir("/Users/fdangelo/PycharmProjects/myRBM/")

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self._current_day = datetime.datetime.now().strftime("%d%m")
    self._current_time = datetime.datetime.now().strftime("%H%M%S")
    self._log_dir = 'CVAElogs/scalars/' + datetime.datetime.now().strftime("%d%m") + '/' + datetime.datetime.now().strftime(
        "%H%M%S") + '/train'
    self._file_writer = tf.summary.create_file_writer(self._log_dir)
    self._file_writer.set_as_default()
    self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(32*32+9,)),
            #tf.keras.layers.Dense(512, activation = 'relu'),
            #tf.keras.layers.Dense(256, activation = 'relu'),
            # No activation, half of the dimensions for the means of the gaussian and half for the sigmas
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim+9,)),
            #tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
            #tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=32*32, activation='linear'),
        ]
    )

  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits


optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  y = x[:,32*32:]
  z_conc = tf.concat(axis=1, values=[z, y])
  x_logit = model.decode(z_conc)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:,:32*32])
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))


def generate_and_save_images(model, epoch, test_input,folder):
  predictions = model.sample(test_input)
  binary_predictions = np.random.binomial(1, predictions.numpy())
  fig = plt.figure(figsize=(3,3))

  for i in range(predictions.shape[0]):
      plt.subplot(3, 3, i+1)
      plt.imshow(tf.reshape(binary_predictions[i, :],(32,32)), cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(folder+'/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()


if __name__ == '__main__':
    #Load and preprocess data
    here = os.path.dirname(os.path.abspath(__file__))
    print(os.getcwd() )
    datah5 = dd.io.load('/cluster/home/fdangelo/RBM_ising/Restricted-Boltzmann-Machines/data/ising/ising_data_L32_large.hdf5')

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
    one_hot_labels = np.zeros((len(class_labels), len(class_names)))
    one_hot_labels[np.arange(len(class_labels)), class_labels] = 1

    data = data_bin[class_names[0]]
    for temperature in class_names[1:]:
        data = np.concatenate([data, data_bin[temperature]])

    # create dictionary for training
    x_train, x_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.1, random_state=42)

    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    x_train_conc = tf.concat(axis=1, values=[x_train, y_train])
    x_test_conc = tf.concat(axis=1, values=[x_test, y_test])

    TRAIN_BUF = x_train.shape[0]
    BATCH_SIZE = 64

    TEST_BUF = x_test.shape[0]

    #we can use tf.data to create batches and shuffle the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train_conc).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test_conc).shuffle(TEST_BUF).batch(BATCH_SIZE)

    epochs = 200
    latent_dim = 50
    num_examples_to_generate = 9

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
    random_c = np.identity(9)
    random_vector_for_generation = tf.concat(axis=1, values=[random_vector_for_generation, random_c])
    model = CVAE(latent_dim)
    results_path = 'results/'+ model._current_day+'/'+model._current_time
    model_path = 'models/'+ model._current_day
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generate_and_save_images(model, 0, random_vector_for_generation, results_path)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(optimizer, gradients, model.trainable_variables)
        end_time = time.time()


        if epoch % 1 == 0:
            model.save_weights(model_path+'/model_weights_'+model._current_time+'.h5' )
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            with tf.name_scope('Performance Metrics'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
              tf.summary.scalar('ELBO', elbo, step = epoch)
            generate_and_save_images(
                model, epoch, random_vector_for_generation,results_path)
