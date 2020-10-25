import tensorflow as tf
import numpy as np
import datetime
import math
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import deepdish as dd
from sklearn.neighbors import NearestNeighbors
import yaml
import os
import time

class RBM():
    def __init__(self, visible_dim, hidden_dim, number_of_epochs, picture_shape, batch_size, training_algorithm='cd',
                 initializer='glorot', k=1, n_test_samples=500, l_1=-1E-6, pcd_restart = 200):
        self._n_epoch = number_of_epochs
        self._v_dim = visible_dim
        self._h_dim = hidden_dim
        self._batch_size = batch_size
        self._picture_shape = picture_shape
        self.n_test_samples = n_test_samples
        self.training_algorithm = training_algorithm
        self.epoch = 1
        self.pcd_restart = pcd_restart
        self.initializer = initializer
        self.k = k
        self.l_1 = l_1
        self.model = self.model()
        self._current_time = datetime.datetime.now().strftime("%d%m-%H%M%S")
        self.d_m_folder = 'results/models/' + datetime.datetime.now().strftime("%d%m")
        self._log_dir = 'logs/scalars/' + datetime.datetime.now().strftime(
            "%d%m") + '/' + datetime.datetime.now().strftime("%H%M%S") + '/train'
        self._file_writer = tf.summary.create_file_writer(self._log_dir)
        self._file_writer.set_as_default()

    # @tf.function
    def model(self):
        if self.initializer == 'glorot':
            self.weights = tf.Variable(
                tf.random.normal([self._h_dim, self._v_dim], mean=0.0, stddev=0.1, dtype=tf.float64) * tf.cast(tf.sqrt(
                    2 / (self._h_dim + self._v_dim)), tf.float64),
                tf.float64, name="weights")
        elif self.initializer == 'normal':
            self.weights = tf.Variable(
                tf.random.normal([self._h_dim, self._v_dim], mean=0.0, stddev=0.1,  dtype=tf.float64),
                tf.float64, name="weights")
        self.visible_biases = tf.Variable(tf.random.uniform([1, self._v_dim], 0, 0.1,  dtype=tf.float64),
                                          tf.float64, name="visible_biases")
        self.hidden_biases = tf.Variable(tf.random.uniform([1, self._h_dim], 0, 0.1, dtype=tf.float64),
                                         tf.float64, name="hidden_biases")
        self.model_dict = {'weights': self.weights, 'visible_biases': self.visible_biases,
                           'hidden_biases': self.hidden_biases}
        return

    def update_model(self):
        for key, value in self.model_dict.items():
            setattr(self, key, value)

    def save_model(self):
        """
        Save the current RBM model as .h5 file dictionary with  keys: {'weights', 'visible_biases', 'hidden_biases' }
        """
        model_dict_save = {'weights': self.weights.numpy(), 'visible_biases': self.visible_biases.numpy(),
                           'hidden_biases': self.hidden_biases.numpy()}
        if not os.path.exists(self.d_m_folder):
            os.makedirs(self.d_m_folder)
        return dd.io.save(self.d_m_folder + '/' + self._current_time + 'model.h5', model_dict_save)

    def save_param(self, optimizer, data=None):
        to_save = {}
        to_save = {**to_save, **optimizer.__dict__}
        del to_save['machine']
        if data is not None:
            to_save['data'] = data

        variables = self.__dict__
        not_save = ['_file_writer', 'model', 'visible_biases', 'hidden_biases', 'weights', 'model_dict']
        for key, value in variables.items():
            if key not in not_save:
                to_save[key] = value
        with open('results/models/' + self._current_time + 'parameters.yml', 'w') as yaml_file:
            yaml.dump(to_save, stream=yaml_file, default_flow_style=False)
        # display parameters in tensorboard
        as_text_matrix = [[k, str(w)] for k, w in sorted(to_save.items())]
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(as_text_matrix), step=1)

    def from_saved_model(self, model_path):
        """
        Build a model from the saved parameters.

        :param model_path: string
                           path of .h5 file containing dictionary of the model with  keys: {'weights', 'visible_biases', 'hidden_biases' }

        :return: loaded model
        """

        model_dict = dd.io.load(model_path)
        self.weights = model_dict['weights'].astype(np.float64)
        self.visible_biases = model_dict['visible_biases'].astype(np.float64)
        self.hidden_biases = model_dict['hidden_biases'].astype(np.float64)

        return self

    def calculate_state(self, probability):
        """
        Given the probability(x'=1|W,b) = sigmoid(Wx+b) computes the next state by sampling from the relative binomial distribution.
        x and x' can be the visible and hidden layers respectively or viceversa.

        :param probability: array, shape(visible_dim) or shape(hidden_dim)

        :return: array , shape(visible_dim) or shape(hidden_dim)
                 0,1 state of each unit in the layer
        """

        s = np.random.binomial(1, probability)
        return s.astype(np.float64)

    def forward(self, batch,beta = 1):
        if beta == 1:
            beta = np.ones((batch.shape[0]))
        hidden_probabilities = tf.sigmoid(beta*tf.tensordot(batch, self.weights, axes=[[1], [
            1]]) + beta*self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states = self.calculate_state(hidden_probabilities)
        return hidden_states

    def backward(self,hidden_batch, beta = 1):
        visible_probabilities = tf.sigmoid(beta*tf.tensordot(hidden_batch, self.weights, axes=[[1], [
            0]]) + beta*self.visible_biases)  # dimension W + 1 row for biases
        visible_states = self.calculate_state(visible_probabilities)
        return visible_states


    def sample(self, inpt=[], n_step_MC=1, p_0=0.5, p_1=0.5):
        """
        Sample from the RBM with n_step_MC steps markov chain.

        :param inpt: array shape(visible_dim), It is possible to start the markov chain from a given point from the dataset or from a random state

        :param n_step_MC: scalar, number of markov chain steps to sample.

        :return: visible_states_1: array shape(visible_dim) visible state after n_step_MC steps

                 visible_probabilities_1: array shape(visible_dim) probabilities from which visible_states_1 is sampled

                 inpt: array shape(picture), return the tarting point of the markov chain

                 evolution_MC, list containing all the states of the markov chain till the sample
        """
        if len(inpt) == 0:
            # inpt = tf.constant(np.random.randint(2, size=self._v_dim), tf.float32)
            inpt = tf.constant(np.random.choice([0, 1], size=self._v_dim, p=[p_0, p_1]), tf.float64)
        hidden_probabilities_0 = tf.sigmoid(
            tf.add(tf.tensordot(self.weights, inpt, 1), self.hidden_biases))  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        evolution_MC = [inpt]
        for _ in range(n_step_MC):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0, self.weights, 1),
                                                        self.visible_biases))  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights), 1),
                                                       self.hidden_biases))  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            evolution_MC.append(visible_states_1.reshape(self._v_dim, ))
        return visible_states_1, visible_probabilities_1, inpt, evolution_MC

    def parallel_sample(self, inpt=[], n_step_MC=1, p_0=0.5, p_1=0.5, n_chains=1, save_evolution=False):
        if len(inpt) == 0:
            inpt = np.random.choice([0, 1], size=(n_chains, self._v_dim), p=[p_0, p_1]).astype(np.float64)
        else:
            # check shape
            if len(inpt.shape) != 2:
                inpt = inpt.reshape(1, inpt.shape[0])
        if save_evolution:
            evolution = np.empty((n_step_MC,inpt.shape[0],self._v_dim))
            evolution[0,:,:] = inpt
        hidden_probabilities_0 = tf.sigmoid(
            tf.tensordot(inpt, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        for i in range(n_step_MC):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [
                0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [
                1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            if save_evolution:
                evolution[i] = visible_states_1
        if save_evolution:
            return visible_states_1, visible_probabilities_1, inpt, evolution
        else:
            return visible_states_1, visible_probabilities_1, inpt


    def parallel_sample_beta(self, inpt=[], n_step_MC=1, p_0=0.5, p_1=0.5, n_chains=1, beta = 1):
        if len(inpt) == 0:
            inpt = np.random.choice([0, 1], size=(n_chains, self._v_dim), p=[p_0, p_1]).astype(np.float64)
        else:
            # check shape
            if len(inpt.shape) != 2:
                inpt = inpt.reshape(1, inpt.shape[0])
        hidden_states_0 = self.forward(inpt,beta)
        for i in range(n_step_MC):
            visible_states_1 = self.backward(hidden_states_0)
            hidden_states_1 = self.forward(visible_states_1)
            hidden_states_0 = hidden_states_1
        return visible_states_1, hidden_states_1, inpt

    def contr_divergence(self, data_point, L2_l=0):
        """
        Perform contrastive divergence given a data point.

        :param data_point: array, shape(visible layer)
                           data point sampled from the batch

        :param L2_l: float, lambda for L2 regularization, default = 0 so no regularization performed

        :return: delta_w: array shape(hidden_dim, visible_dim)
                          Array of the same shape of the weight matrix which entries are the gradients dw_{ij}

                 delta_vb: array, shape(visible_dim)
                           Array of the same shape of the visible_biases which entries are the gradients d_vb_i

                 delta_vb: array, shape(hidden_dim)
                           Array of the same shape of the hidden_biases which entries are the gradients d_hb_i

        """
        hidden_probabilities_0 = tf.sigmoid(
            tf.add(tf.tensordot(self.weights, data_point, 1), self.hidden_biases))  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0
        for _ in range(self.k):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0, self.weights, 1),
                                                        self.visible_biases))  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights), 1),
                                                       self.hidden_biases))  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.reshape(tf.tensordot(hidden_states_0_copy, data_point, 0), (self._h_dim, self._v_dim))
        vh_1 = tf.reshape(tf.tensordot(hidden_states_1, visible_states_1, 0), (self._h_dim, self._v_dim))
        delta_w = tf.add(vh_0, - vh_1) + L2_l * self.weights
        delta_vb = tf.add(data_point, - visible_states_1) + L2_l * self.visible_biases
        delta_hb = tf.add(hidden_states_0_copy, - hidden_states_1) + L2_l * self.hidden_biases
        return delta_w.numpy(), delta_vb.numpy(), delta_hb.numpy()  # , visible_states_1

    # @tf.function
    def parallel_cd(self, batch):
        hidden_probabilities_0 = tf.sigmoid(
            tf.tensordot(batch, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0.copy()
        for _ in range(self.k):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [
                0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [
                1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.tensordot(hidden_probabilities_0, batch, axes=[[0], [0]])
        vh_1 = tf.tensordot(hidden_probabilities_1, visible_states_1, axes=[[0], [0]])
        delta_w = (vh_0 - vh_1) / batch.shape[0]
        delta_vb = np.average(batch - visible_states_1, 0)
        delta_hb = np.average(hidden_probabilities_0 - hidden_probabilities_1, 0)

        # return delta_w.numpy(), delta_vb, delta_hb
        return delta_w.numpy() + self.l_1 * np.sign(self.weights.numpy()), delta_vb + self.l_1 * np.sign(
            self.visible_biases.numpy()), delta_hb + self.l_1 * np.sign(self.hidden_biases.numpy())

    def parallel_pcd(self, batch, initial_vis,init_hidden_prob):
        hidden_probabilities_0 = tf.sigmoid(
            tf.tensordot(batch, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0.copy()
        for _ in range(self.k):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [
                0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [
                1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.tensordot(init_hidden_prob, initial_vis, axes=[[0], [0]])
        vh_1 = tf.tensordot(hidden_probabilities_1, visible_states_1, axes=[[0], [0]])
        delta_w = (vh_0 - vh_1) / batch.shape[0]
        delta_vb = np.average(initial_vis - visible_states_1[:initial_vis.shape[0],:], 0)
        delta_hb = np.average(init_hidden_prob - hidden_probabilities_1[:initial_vis.shape[0],:], 0)

        # return delta_w.numpy(), delta_vb, delta_hb
        return delta_w.numpy() + self.l_1 * np.sign(self.weights.numpy()), delta_vb + self.l_1 * np.sign(
            self.visible_biases.numpy()), delta_hb + self.l_1 * np.sign(self.hidden_biases.numpy()), visible_states_1

    def parallel_tempering(self,batch,n_beta):
        beta = np.linspace(0, 1, n_beta)
        self.parallel_sample_beta(inpt = batch, beta = b)


    def energy(self, visible_config):
        hidden_probabilities = tf.sigmoid(
            tf.tensordot(visible_config, self.weights, axes=[[1], [1]]) + self.hidden_biases)
        hidden_state = self.calculate_state(hidden_probabilities)

        E = -tf.reshape(tf.tensordot(visible_config, self.visible_biases, axes = [[1],[1]]), [-1])-tf.reshape(tf.tensordot(hidden_state, self.hidden_biases, axes = [[1],[1]]),[-1]) \
            - tf.linalg.tensor_diag_part(tf.matmul(hidden_state,  tf.transpose(tf.tensordot(visible_config, self.weights, axes=[[1], [1]]))))

        return E

    def clamped_free_energy(self,test_point,mean = True):
        """
        N.B. the free energy is defined with the minus in front
        Compute the free energy of the RBM, it is useful to compute the pseudologlikelihood.
        F(v) = - log \sum_h e^{-E(v,h)} = -bv - \sum_i \log(1 + e^{c_i + W_i v}) where v= visible state, h = hidden state,
        b = visible_biases, c = hidden_biases, W_i = i-th column of the weights matrix
        :param test_point: array, shape(visible_dim)
                           random point sampled from the test set
        :return: scalar
        """
        bv = tf.tensordot(test_point, tf.transpose(self.visible_biases), 1)
        wx_b = tf.tensordot(test_point, self.weights, axes=[[1], [1]]) + self.hidden_biases
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.math.exp(wx_b)), 1)

        if mean:
            return tf.math.reduce_mean(-hidden_term - tf.reshape(bv, [-1]))
        else:
            return -hidden_term - tf.reshape(bv, [-1])

    def exact_partition(self,lista):
        lista_tf = tf.convert_to_tensor(
            lista,
            dtype=tf.float64)
        Z = 0
        for i, conf in enumerate(lista_tf):
            p_v = tf.exp(tf.tensordot(self.hidden_biases, conf, axes=[[1], [1]])) * tf.reduce_prod(
                (1 + tf.exp(tf.tensordot(conf, self.weights, axes=[[1], [0]]) + self.visible_biases)),
                1)
            Z += tf.reduce_sum(p_v)
        return Z

    def exact_log_likelihood(self, points, lista):
        exact_log_Z = tf.math.log(self.exact_partition(lista))
        log_L = - self.clamped_free_energy(points)
        return log_L - exact_log_Z, exact_log_Z

    def variable_summaries(self, var, step):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, step)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, step)
            tf.summary.scalar('max', tf.reduce_max(var), step)
            tf.summary.scalar('min', tf.reduce_min(var), step)
            tf.summary.histogram('histogram', var, step=step)

    def train(self, data, optimizer, metric_monitor):
        """
        This function shuffle the dataset and create #data_train/batch_size mini batches and perform contrastive divergence on each vector of the batch.
        The upgrade of the parameters is performed only at the end of each batch by taking the average of the gradients on the batch.
        In the last part a random datapoint is sampled from the test set to calculate the error reconstruction. The entire procedure is repeted
         _n_epochs times.

        :param data: dict, dictionary of numpy arrays with labels ['x_train','y_train','x_test', 'y_test']
               optimizer: object optimizer

        :return: self
        """
        print('Start training...')

        # config = np.array([i for i in product(range(2), repeat=self._h_dim)]).astype(np.int32)
        # lista = np.split(config, 256)
        # config = None
        test_fixed = np.random.randint(low=0, high=data['x_test'].shape[0], size=self.n_test_samples)
        for epoch in range(1, self._n_epoch + 1):
            start_time = time.time()
            self.epoch = epoch
            # sys.stdout.write('\r')
            print('Model',self._current_time,'Epoch:', epoch, '/', self._n_epoch)
            np.random.shuffle(data['x_train'])
            for i in tqdm(range(0, data['x_train'].shape[0], self._batch_size)):

                if self.training_algorithm == 'cd':
                    x_train_mini = data['x_train'][i:i + self._batch_size]
                    batch_dw, batch_dvb, batch_dhb = self.parallel_cd(x_train_mini)


                elif self.training_algorithm == 'pcd':
                    from_noise = True
                    initial_vis = data['x_train'][i:i + self._batch_size]
                    init_hidden_prob = tf.sigmoid(
                        tf.tensordot(initial_vis, self.weights, axes=[[1], [1]]) + self.hidden_biases)
                    if (not epoch % self.pcd_restart or epoch == 1) and i == 0: #when restart the chain
                        if from_noise:
                            x_train_mini = np.random.randint(2, size=(self._batch_size, self._v_dim)).astype(np.float64)
                        else:
                            x_train_mini = data['x_train'][i:i + self._batch_size]
                    batch_dw, batch_dvb, batch_dhb, x_train_mini = self.parallel_pcd(x_train_mini, initial_vis,init_hidden_prob[:initial_vis.shape[0],:])

                self.grad_dict = {'weights': batch_dw,
                                  'visible_biases': batch_dvb,
                                  'hidden_biases': batch_dhb}
                optimizer.fit()
            # Save model every epoch
            self.save_model()

            # test every epoch
            metric_monitor.fit(data['x_test'])
            end_time = time.time()
            print('time:', end_time - start_time)

