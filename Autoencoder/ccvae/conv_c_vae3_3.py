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
from measuring_temperature_final import create_conv_model, create_conv_T_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from IPython import display
import io
import tensorflow as tf
import argparse
import yaml 
from ising_estimations import calcEnergy, correlation_function, correlation_lenght

#os.chdir("/Users/fdangelo/PycharmProjects/myRBM/")

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim,num_classes,config):
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
          tf.keras.layers.Conv2D(32, (config.filter_size_in, config.filter_size_in), activation=config.activation, padding="same", input_shape=(32, 32, 1)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D((2, 2)),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv2D(64, (config.filter_size_in, config.filter_size_in), padding="same", activation=config.activation),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D((config.filter_size_in, config.filter_size_in)),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv2D(64, (config.filter_size_in, config.filter_size_in), padding="same", activation=config.activation),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(latent_dim + latent_dim)
  		  ]
	 )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim+num_classes,)),
          tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
          tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=config.filter_size_gen,strides=(1, 1),padding="SAME",activation=config.activation),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=config.filter_size_gen,strides=(1, 1),padding="SAME",activation=config.activation),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.UpSampling2D(size=(2, 2)),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=config.filter_size_gen,strides=(1, 1),padding="SAME",activation=config.activation),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=config.filter_size_gen, strides=(1, 1), padding="SAME"),
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
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
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

def correlation(corr_samples,corr_data,class_names,colors, legend):
    radii = [0,1,2,3,4,5,6,8,10]
    figure = plt.figure(figsize=(10, 10))
    for i in range(len(class_names)):
        plt.plot(radii,corr_samples[i,:], color=colors[i])
        plt.scatter(radii,corr_data[i,:], color=colors[i],label=str(legend[i]))
    plt.title('Correlation')
    plt.xlabel('Radii')
    plt.ylabel('Correlations')
    plt.grid()
    plt.legend()
    return plot_to_image(figure)

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


def scatter(embedding, labels, colors=[], names=[], save=False):
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    if len(colors) == 0:
        colors = get_colors(len(labels))
    if len(names) == 0:
        names = np.unique(labels)
    figure = plt.figure(figsize=(10, 10))

    if len(labels) != len(embedding):
        raise ValueError("Number of labels does not match number of embeddings")

    for label in range(np.unique(labels).shape[0]):
        idx = (labels == label)
        plt.scatter(embedding[idx, 0], embedding[idx, 1], s=15, c=colors[label], alpha=0.6, label=str(names[label]))

    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 0)
    plt.ylabel("E(T)")
    plt.xlabel("M(T)")
    plt.grid()
    return plot_to_image(figure)

def scatter_2(data,samples,temps,title, save=False):
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(temps, samples, s=400, c='blue',marker='o',alpha=0.8, label='Samples') 
    plt.scatter(temps, data, s=400, c='red', marker='^',alpha=0.8, label='Data') 


    plt.legend()
    #plt.xlim(-1, 1)
    #plt.ylim(-1, 0)
    plt.title(title,fontsize=30)
    plt.xlabel("T")
    plt.grid()
    plt.legend(fontsize=20)
    plt.show()
    return plot_to_image(figure)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

if __name__ == '__main__':
    #Load and preprocess data
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filter_size_in', type=int, metavar='N', default=3,
                        help='Convolutional filter size encoder %(default)s.')
    parser.add_argument('--filter_size_gen', type=int, metavar='N', default=3,
                        help='Convolutional filter size decoder %(default)s.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=200,
                    help='Dimension of latent space %(default)s.')
    parser.add_argument('--epochs', type=int, metavar='N', default=5000,
                    help='Number of epochs %(default)s.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate of optimizer(s). Default: ' +
                             '%(default)s.')
    parser.add_argument('--activation', type=str, default='relu',
                    help='Activation function: ' +
                         '%(default)s.')
    args = parser.parse_args()

    #Load and preprocess data
    dictionary_parameters = vars(args)

    here = os.path.dirname(os.path.abspath(__file__))

    datah5 = dd.io.load('/cluster/scratch/fdangelo/RBM/data/ising/ising_data_heterogeneousJmu1sigma1_L32_new.hdf5')


    # Transform -1 in 0 and take spin up as standard configuration
    binarizer = Binarizer(threshold=0)
    keys = list(datah5.keys())
    # put here the temperature from keys that you want to use for the training
    #class_names = [keys[i] for i in [4, 6, 7, 8, 9, 10, 11, 12, 16]]
    class_names = [keys[i] for i in [0,2,4,6,8,12]]
    print(class_names)
    radii = [0, 1, 2, 3, 4, 5, 6, 8, 10]
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

    correlation_r_data_T = np.zeros((len(class_names), len(radii)))
    # class labels even if they are not really useful here
    class_labels = np.asarray(
        list(itertools.chain.from_iterable(itertools.repeat(x, n_samples) for x in range(0, len(class_names)))))
    one_hot_labels = np.zeros((len(class_labels), len(class_names)))
    one_hot_labels[np.arange(len(class_labels)), class_labels] = 1

    data = data_bin[class_names[0]]
    ham_data = np.mean([calcEnergy(np.array(slice).reshape((32,32))) for slice in datah5[class_names[0]]])
    for temperature in class_names[1:]:
       data = np.concatenate([data, data_bin[temperature]])

    # create dictionary for training
    x_train, x_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.1, random_state=42)
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 1)).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 1)).astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    #x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    #x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    #x_train_conc = tf.concat(axis=1, values=[x_train, y_train])
    #x_test_conc = tf.concat(axis=1, values=[x_test, y_test])

    temperatures = [round(float(i.replace('T=','')),3) for i in class_names]
    new_colors2 = np.asarray(["#cacd30", "#1449bb", "#427700", "#9d0575", "#ff8c5d", "#ecacff", "#ff5474"])
    temps_plot = [i[:-4] for i in class_names]

    TRAIN_BUF = x_train.shape[0]
    BATCH_SIZE = 128

    TEST_BUF = x_test.shape[0]

    #we can use tf.data to create batches and shuffle the dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(TEST_BUF).batch(BATCH_SIZE)

    epochs = args.epochs
    latent_dim = args.latent_dim
    num_examples_to_generate = len(class_names)

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.

    model = CVAE(latent_dim,len(class_names), args)
    optimizer = tf.keras.optimizers.Adam(args.lr)


    classifier = create_conv_T_model(len(class_names))
    saved = '/cluster/scratch/fdangelo/RBM/measuring_temp/measuring_temp_logs/checkpoints/cp1104-100113.ckpt'
    classifier.load_weights(saved)

    correlation_r_data_T = np.load('/cluster/scratch/fdangelo/RBM/data/ising/correlation_r_data_T6_temp_J=1.npy')
    correlation_r_data_T = correlation_r_data_T[:,:9]
    energy_data = np.mean(np.load('/cluster/scratch/fdangelo/RBM/data/ising/Energy_data_T6_temp_J=1.npy'),0)
    magn_data = np.mean(np.load('/cluster/scratch/fdangelo/RBM/data/ising/magn_data_T6_temp_J=1.npy'),0)

    results_path = 'results/'+ model._current_day+'/'+model._current_time
    model_path = 'models/'+ model._current_day
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + 'parameters.yml', 'w') as yaml_file:
        yaml.dump(dictionary_parameters, stream=yaml_file, default_flow_style=False)

    # display parameters in tensorboard
    as_text_matrix = [[k, str(w)] for k, w in sorted(dictionary_parameters.items())]
    config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(as_text_matrix), step=1)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, train_y in train_dataset:
            compute_apply_gradients(model, train_x, train_y,optimizer)
        end_time = time.time()


        if epoch % 50 == 0 or epoch == 1:
            model.save_weights(model_path+'/model_weights_'+model._current_time+'epoch'+str(epoch)+'.h5' )
            loss = tf.keras.metrics.Mean()
            loss_2 = tf.keras.metrics.Mean()
            loss_3 = tf.keras.metrics.Mean()
            lista = []

            for test_x, test_y in test_dataset:
                loss(compute_loss(model, test_x, test_y))

                random_vector_for_generation = tf.random.normal(shape=[test_x.shape[0], latent_dim])
                predictions = model.sample(test_y,random_vector_for_generation)
                binary_predictions = np.random.binomial(1, predictions.numpy())
                lista.append(binary_predictions)
                kl, kl_inv = KL_divergence(binary_predictions,test_x,10,BATCH_SIZE)
                loss_2(kl)
                loss_3(kl_inv)
            elbo = -loss.result()
            kl_monitor = loss_2.result()
            kl_inv_monitor = loss_3.result()
            display.clear_output(wait=False)

            test_samples = np.concatenate(lista)
            test_samples = test_samples.reshape(-1, 32, 32, 1)
            prediction = classifier.predict(test_samples)
            predicted_labels = np.argmax(prediction, axis=1)
            state_spin = np.squeeze(np.where(test_samples == 0, -1, test_samples))
            print(state_spin.shape)
            magn_samples = [np.mean(state_spin[i]) for i in range(state_spin.shape[0])]
            En_samples = [calcEnergy(slice.reshape((32, 32))) for slice in state_spin]
            correlation_r_samples_T = np.zeros((len(class_names), len(radii)))
            
            mean_magn = [] 
            mean_hamiltonian_samples = [] 

            for idx1, key in enumerate(class_names):
                mean_magn.append(np.mean(np.asarray(magn_samples)[predicted_labels==idx1]))
                mean_hamiltonian_samples.append(np.mean(np.asarray(En_samples)[predicted_labels==idx1]))
                for idx2, r in enumerate(radii):
                    correlation_r_samples_T[idx1][idx2] = np.mean(
                        [correlation_function(slice, r) for slice in state_spin[predicted_labels == idx1]])

            twoD_emb_samples = np.asarray([magn_samples, En_samples]).T


            scatter_plot = scatter(twoD_emb_samples, predicted_labels, new_colors2, temps_plot, save=True)

            correlation_plot = correlation(correlation_r_samples_T, correlation_r_data_T, class_names, new_colors2,
                                           temps_plot)
            magn_plot = scatter_2(magn_data,mean_magn,temperatures, 'Magnetization')
            energy_plot = scatter_2(energy_data, mean_hamiltonian_samples,temperatures, 'Energy')

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
            #tf.summary.image('Data', tf.cast(tf.reshape(data_bin[class_names[0]][i],(len(class_names),32,32,1)),dtype = tf.float32), max_outputs=10000, step=epoch)
            #pic = generate_and_save_images(model)
            #tf.summary.image('Samples',pic,max_outputs=10000,step = epoch)
            tf.summary.image("Scatter", scatter_plot, step=epoch)
            tf.summary.image("Correlation", correlation_plot, step=epoch)
            tf.summary.image("Magnetization", magn_plot, step=epoch)
            tf.summary.image("Energy", energy_plot, step=epoch)
            #with tf.name_scope('Weights'):
                #variable_summaries(model.weights, step = epoch)
            #if not epoch%5:
                #error_correlation = correlation(model,class_names,correlation_r_data_T,epoch,results_path)
                #with tf.name_scope('Correlation error'):
                    #tf.summary.scalar('Correlation error', error_correlation, step = epoch)