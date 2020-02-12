from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import numpy as np
import glob
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
#import PIL
import datetime
#import imageio
import deepdish as dd
import itertools
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from IPython import display
import tensorflow as tf
#from ising_estimations import calcEnergy, correlation_function, correlation_lenght

#os.chdir("/Users/fdangelo/PycharmProjects/myRBM/")

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim,num_classes):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.num_classes = num_classes
    self._current_day = datetime.datetime.now().strftime("%d%m")
    self._current_time = datetime.datetime.now().strftime("%H%M%S")
    self._log_dir = 'CVAElogs/scalars/' + datetime.datetime.now().strftime("%d%m") + '/' + datetime.datetime.now().strftime(
        "%H%M%S") + '/train'
    self._file_writer = tf.summary.create_file_writer(self._log_dir)
    self._file_writer.set_as_default()

    self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(32*32,)),
            tf.keras.layers.Dense(768, activation='relu'),
            tf.keras.layers.Dense(512, activation = 'relu'),
            #tf.keras.layers.Dense(256, activation = 'relu'),
            # No activation, half of the dimensions for the means of the gaussian and half for the sigmas
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim+self.num_classes,)),
            #tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=768, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=32*32, activation='linear'),
        ]
    )


  def sample(self, label, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, label, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, labels, apply_sigmoid=False):
    z_tilde = tf.concat(axis=1, values=[z, labels])
    logits = self.generative_net(z_tilde)
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


def compute_loss(model, x, labels):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z, labels)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_apply_gradients(model, x, labels, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x,labels)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def recon_c_e( data,samples_prob):
    loss = tf.keras.backend.binary_crossentropy(data, samples_prob).numpy()
    loss = np.sum(loss, 1)
    return np.average(loss)

def state_spins(states):
    states_spins = np.array([np.where(slice==0,-1,slice) for slice in states])
    return states_spins

def correlation(model,class_names,corr_data,epoch,folder):
    num_examples_to_generate = 1000
    matplotlib.use('Agg')
    radii = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    colors = ['blue','red','orange']
    correlation_r_cvae_T = np.zeros((model.num_classes, len(radii)))
    for enum, m in enumerate(class_names):
        random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
        temperature = np.zeros((num_examples_to_generate, len(class_names)))
        temperature[:, enum] = 1
        random_vector_for_generation = tf.concat(axis=1, values=[random_vector_for_generation, temperature])
        predictions = model.sample(random_vector_for_generation)
        binary_predictions = np.random.binomial(1, predictions.numpy())
        for idx2, r in enumerate(radii):
            correlation_r_cvae_T[enum][idx2] = np.mean(
                [correlation_function(state_spins(np.array(slice)[:1024]).reshape(32, 32), r) for slice in
                 binary_predictions])
    for i in range(len(class_names)):
        plt.plot(radii,correlation_r_cvae_T[i,:], color=colors[i])
        plt.plot(radii,corr_data[i,:], '--', color=colors[i])
    plt.title('Correlation')
    plt.xlabel('Radii')
    plt.ylabel('Correlations')
    plt.legend(['CVAE T = 1', 'data T = 1', 'CVAE T = 2.5','data T = 2.5', 'CVAE T = 4.0', 'data T=4.0'], loc=4)
    plt.savefig(folder + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    return np.mean(np.abs(np.divide(correlation_r_cvae_T-corr_data,corr_data)))

def KL_divergence(samples,data,k_neigh,batch_size):
    #todo: I should try with reconstructing point starting from other points
    #rnd_test_points_idx = np.random.randint(low=0, high=data.shape[0], size=n_points)
    # check the shape
    test_points = data.numpy()
    if len(samples.shape)!=2: 
      samples = samples.reshape(batch_size,-1)
    if len(data.shape)!=2: 
      test_points = test_points.reshape(batch_size,-1)
    n_points = data.shape[0]
    nbrs_data = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
    nbrs_data.fit(test_points)
    nbrs_model = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
    nbrs_model.fit(samples)

    rho, _ = nbrs_data.kneighbors(test_points)
    nu, _ = nbrs_model.kneighbors(test_points)

    rho_inv, _ = nbrs_data.kneighbors(samples)
    nu_inv, _ = nbrs_model.kneighbors(samples)

    l = 0
    l_inv = 0
    # -2 is needed because in rho the first distance is always 0 and then with the point itself that we should not consider,
    #to effectively pick the k-th neigh w.r.t test points and reconstructions we have to take the k-th in rho and the k-th -1 in nu.
    for i in range(n_points):
        l += np.log(nu[i, k_neigh-2] / rho[i, k_neigh-1])
        l_inv += np.log(rho_inv[i, k_neigh-2] / nu_inv[i, k_neigh-1])
    DKL = data.shape[1]/ n_points * l + np.log(n_points / (n_points - 1))
    DKL_inv = data.shape[1]/ n_points * l_inv + np.log(n_points / (n_points - 1))
    return DKL, DKL_inv

def generate_and_save_images(model):
    random_vector_for_generation = tf.random.normal(shape=[model.num_classes, model.latent_dim])
    labels = np.identity(len(class_names))
    predictions = model.sample(labels,random_vector_for_generation)
    binary_predictions = np.random.binomial(1, predictions.numpy())
    pic = tf.reshape(tf.convert_to_tensor(binary_predictions,dtype=tf.float32),(model.num_classes,32,32,1))
    return pic

def variable_summaries(var, step):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, step)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, step)
        tf.summary.scalar('max', tf.reduce_max(var), step)
        tf.summary.scalar('min', tf.reduce_min(var), step)
        tf.summary.histogram('histogram', var, step = step)

if __name__ == '__main__':
    #Load and preprocess data
    here = os.path.dirname(os.path.abspath(__file__))

    datah5 = dd.io.load('/Users/fdangelo/PycharmProjects/myRBM/data/ising/ising_data_L32_large.hdf5')


    # Transform -1 in 0 and take spin up as standard configuration
    binarizer = Binarizer(threshold=0)
    keys = list(datah5.keys())
    # put here the temperature from keys that you want to use for the training
    #class_names = [keys[i] for i in [4, 6, 7, 8, 9, 10, 11, 12, 16]]
    class_names = [keys[i] for i in [6,8,10,11,12,16]]
    n_samples = datah5[keys[0]].shape[0]
    datah5_norm = {}
    data_bin = {}
    for key in class_names:
      if key==keys[9]:
        datah5_norm[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
        data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5_norm[key]])
      else:
        #datah5_norm[key] = np.array([np.where(np.sum(slice)<0,-slice,slice) for slice in datah5[key]])
        data_bin[key] = np.array([binarizer.fit_transform(slice) for slice in datah5[key]])

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
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    #x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    #x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    #x_train_conc = tf.concat(axis=1, values=[x_train, y_train])
    #x_test_conc = tf.concat(axis=1, values=[x_test, y_test])

    TRAIN_BUF = x_train.shape[0]
    BATCH_SIZE = 128

    TEST_BUF = x_test.shape[0]

    #we can use tf.data to create batches and shuffle the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(TEST_BUF).batch(BATCH_SIZE)

    epochs = 1000
    latent_dim = 200
    num_examples_to_generate = len(class_names)

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.

    model = CVAE(latent_dim,len(class_names))
    results_path = 'results/'+ model._current_day+'/'+model._current_time
    model_path = 'models/'+ model._current_day
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, train_y in train_dataset:
            compute_apply_gradients(model, train_x, train_y,optimizer)
        end_time = time.time()


        if epoch % 5 == 0 or epoch == 1:
            model.save_weights(model_path+'/model_weights_'+model._current_time+'epoch'+str(epoch)+'.h5' )
            loss = tf.keras.metrics.Mean()
            loss_2 = tf.keras.metrics.Mean()
            loss_3 = tf.keras.metrics.Mean()
            for test_x, test_y in test_dataset:
                loss(compute_loss(model, test_x, test_y))

                random_vector_for_generation = tf.random.normal(shape=[test_x.shape[0], latent_dim])
                predictions = model.sample(test_y,random_vector_for_generation)
                binary_predictions = np.random.binomial(1, predictions.numpy())
                kl, kl_inv = KL_divergence(binary_predictions,test_x,10,BATCH_SIZE)
                loss_2(kl)
                loss_3(kl_inv)
            elbo = -loss.result()
            kl_monitor = loss_2.result()
            kl_inv_monitor = loss_3.result()
            display.clear_output(wait=False)

            print('Model:{}, Epoch: {}, Test set ELBO: {}, KL divegence: {}, ''time elapse for current epoch {}'.format(model._current_time,epoch,
                                                            elbo,kl_monitor,
                                                            end_time - start_time))
            with tf.name_scope('Performance Metrics'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
              tf.summary.scalar('ELBO', elbo, step = epoch)
            with tf.name_scope('Performance Metrics'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
              tf.summary.scalar('KL', kl_monitor, step = epoch)
            with tf.name_scope('Performance Metrics'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
              tf.summary.scalar('KL_inv', kl_inv_monitor, step = epoch)

            i = np.random.randint(0,data_bin[class_names[0]].shape[0])
            #data = tf.concat([data_bin[class_names[0]][i],data_bin[class_names[1]][i],data_bin[class_names[2]][i],data_bin[class_names[3]][i],data_bin[class_names[4]][i],data_bin[class_names[5]][i]],data_bin[class_names[6]][i],0)
            #tf.summary.image('Data', tf.cast(tf.reshape(data,(len(class_names),32,32,1)),dtype = tf.float32), max_outputs=10000, step=epoch)
            pic = generate_and_save_images(model)
            tf.summary.image('Samples',pic,max_outputs=10000,step = epoch)
            #with tf.name_scope('Weights'):
                #variable_summaries(model.weights, step = epoch)
            #if not epoch%5:
                #error_correlation = correlation(model,class_names,correlation_r_data_T,epoch,results_path)
                #with tf.name_scope('Correlation error'):
                    #tf.summary.scalar('Correlation error', error_correlation, step = epoch)