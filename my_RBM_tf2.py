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
import multiprocessing.dummy as mp
import yaml

'''
class monitoring():
    def reconstruction_cross_e():4 
    
    def ava_sq_error(): 
    
    def pseudo_log_l():
'''

class RBM():
    def __init__(self, visible_dim, hidden_dim, number_of_epochs, picture_shape, batch_size,  training_algorithm='cd', k = 1, n_test_samples=500, init_learning_rate = 0.8):
        self._n_epoch = number_of_epochs
        self._v_dim = visible_dim
        self._h_dim = hidden_dim
        self._l_r = init_learning_rate
        self._batch_size = batch_size
        self._picture_shape = picture_shape
        self.n_test_samples = n_test_samples
        self.training_algorithm = training_algorithm
        self.epoch = 1
        self.k = k
        self.model = self.model()
        self._current_time = datetime.datetime.now().strftime("%d%m-%H%M%S")
        self._log_dir = 'logs/scalars/' + self._current_time + '/train'
        self._file_writer = tf.summary.create_file_writer(self._log_dir)
        self._file_writer.set_as_default()

    #@tf.function
    def model(self):
        self.visible_biases = tf.Variable(tf.random.uniform([1, self._v_dim], 0, 0.1,seed = 42), tf.float32, name="visible_biases")
        self.hidden_biases = tf.Variable(tf.random.uniform([1, self._h_dim], 0, 0.1, seed = 42), tf.float32, name="hidden_biases")
        self.weights = tf.Variable(tf.random.normal([self._h_dim, self._v_dim], mean = 0.0, stddev = 0.1, seed = 42), tf.float32, name="weights")
        self.model_dict = {'weights': self.weights, 'visible_biases': self.visible_biases, 'hidden_biases': self.hidden_biases}
        return

    def update_model(self):
        for key,value in self.model_dict.items():
            setattr(self, key, value)

    def save_model(self):
        """
        Save the current RBM model as .h5 file dictionary with  keys: {'weights', 'visible_biases', 'hidden_biases' }
        """
        model_dict_save = {'weights': self.weights.numpy(), 'visible_biases': self.visible_biases.numpy(),
                           'hidden_biases': self.hidden_biases.numpy()}
        return dd.io.save('results/models/'+self._current_time+'model.h5', model_dict_save)

    def save_param(self, data = None):
        to_save = {}

        if data is not None :
            to_save['data'] = data

        variables = self.__dict__
        not_save = ['_file_writer', 'model', 'visible_biases', 'hidden_biases', 'weights', 'model_dict']
        for key,value in variables.items():
            if key not in not_save:
                to_save[key] = value
        with open('results/models/'+self._current_time+'parameters.yml', 'w') as yaml_file:
            yaml.dump(to_save, stream=yaml_file, default_flow_style=False)

    def from_saved_model(self,model_path):
        """
        Build a model from the saved parameters.

        :param model_path: string
                           path of .h5 file containing dictionary of the model with  keys: {'weights', 'visible_biases', 'hidden_biases' }

        :return: loaded model
        """

        model_dict = dd.io.load(model_path)
        self.weights = model_dict['weights']
        self.visible_biases = model_dict['visible_biases']
        self.hidden_biases = model_dict['hidden_biases']

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
        return s.astype(np.float32)


        #@tf.function
    def sample(self, inpt = [] ,n_step_MC=1,p_0=0.5,p_1=0.5):
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
            #inpt = tf.constant(np.random.randint(2, size=self._v_dim), tf.float32)
            inpt = tf.constant(np.random.choice([0,1], size=self._v_dim,p=[p_0,p_1]), tf.float32)
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, inpt,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        evolution_MC = [inpt]
        for _ in range(n_step_MC): #gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0,self.weights,1), self.visible_biases)) # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            evolution_MC.append(visible_states_1.reshape(self._v_dim,))
        return visible_states_1,visible_probabilities_1,inpt,evolution_MC

    def parallel_sample(self, inpt = [] ,n_step_MC=1,p_0=0.5,p_1=0.5, n_chains = 1, save_evolution = False):
        if len(inpt) == 0:
            inpt = np.random.choice([0, 1], size=(n_chains,self._v_dim), p=[p_0, p_1]).astype(np.float32)
        else:
            #check shape
            if len(inpt.shape) != 2:
                inpt = inpt.reshape(1,inpt.shape[0])
        if save_evolution:
            evolution = np.empty((n_step_MC,self._v_dim))
            evolution[0] = inpt
        hidden_probabilities_0 = tf.sigmoid(tf.tensordot(inpt, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        for i in range(n_step_MC):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
            if save_evolution:
                evolution[i] = visible_states_1
        if save_evolution:
            return visible_states_1, visible_probabilities_1,inpt, evolution
        else:
            return visible_states_1, visible_probabilities_1,inpt

    #@tf.function
    def contr_divergence(self, data_point, L2_l = 0):
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
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, data_point,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0
        for _ in range(self.k): #gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0,self.weights, 1), self.visible_biases))# dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.reshape(tf.tensordot(hidden_states_0_copy, data_point, 0), (self._h_dim,self._v_dim))
        vh_1 = tf.reshape(tf.tensordot(hidden_states_1, visible_states_1, 0), (self._h_dim,self._v_dim))
        delta_w = tf.add(vh_0, - vh_1) +L2_l*self.weights
        delta_vb = tf.add(data_point, - visible_states_1) + L2_l*self.visible_biases
        delta_hb = tf.add(hidden_states_0_copy, - hidden_states_1) + L2_l*self.hidden_biases
        return delta_w.numpy(), delta_vb.numpy(), delta_hb.numpy() #, visible_states_1


    #@tf.function
    def parallel_cd(self, batch):
        hidden_probabilities_0 = tf.sigmoid(tf.tensordot(batch, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0.copy()
        for _ in range(self.k):  # gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.tensordot(hidden_states_0, self.weights, axes=[[1], [0]]) + self.visible_biases)  # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.tensordot(visible_states_1, self.weights, axes=[[1], [1]]) + self.hidden_biases)  # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1

        vh_0 = tf.tensordot(hidden_probabilities_0, batch, axes=[[0], [0]])
        vh_1 = tf.tensordot(hidden_probabilities_1, visible_states_1, axes=[[0], [0]])
        delta_w = (vh_0 - vh_1) / batch.shape[0]
        delta_vb = np.average(batch - visible_states_1, 0)
        delta_hb = np.average(hidden_probabilities_0 - hidden_probabilities_1, 0)

        return delta_w.numpy(), delta_vb, delta_hb

    def energy(self, visible_config):
        hidden_probabilities = tf.sigmoid(tf.add(tf.tensordot(self.weights, visible_config,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_state = self.calculate_state(hidden_probabilities)
        E = -np.inner(visible_config, self.visible_biases) -np.inner(hidden_state,self.hidden_biases) -np.inner(hidden_state, tf.tensordot(self.weights,visible_config,1))

        return E[0]


    def reconstruction_cross_entropy(self,test_points, plot=False):
        """
        Compute the reconstruction cross entropy = - \Sum_[i=1]^d z_i log(p(z_i)) + (1-z_i) log(1-p(z_i)) where i
        is the i-th component of the reconstructed vector and p(z_i) = sigmoid(Wx+b)_i.

        :param test_point: array like
                           Random point sampled from the test set
        :param plot: bool
                    if True plot the reconstruction togheter with the sampled test point for comparison
        :return: scalar
                Reconstruction cross entropy
        """
        
        r_ce_list=[]
        for vec in test_points: 
            reconstruction,prob,_,_ = self.sample(inpt=vec)
            #tf.where is needed to have 0*-\infty = 0
            r_ce = tf.multiply(reconstruction, tf.where(tf.math.is_inf(tf.math.log(prob)),np.zeros_like(tf.math.log(prob)),tf.math.log(prob))) \
                   + tf.multiply((1-reconstruction), tf.where(tf.math.is_inf(tf.math.log(1-prob)),np.zeros_like(tf.math.log(1-prob)), tf.math.log(1-prob)))
            r_ce_list.append(-tf.reduce_sum(r_ce,1)[0])

        if plot:
            reconstruction_plot= self.sample(inpt=test_points[1,:])[0]
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(test_points[1,:].reshape(self._picture_shape),cmap='Greys')
            axes[0].set_title("Original Image")
            axes[1].imshow(np.asarray(reconstruction_plot).reshape(self._picture_shape), cmap='Greys')
            axes[1].set_title("Reconstruction")
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        return np.average(r_ce_list)

    def recon_c_e(self,test_points):
        reconstruction, prob, _ = self.parallel_sample(test_points)
        loss = tf.keras.backend.binary_crossentropy(test_points, prob).numpy()
        loss = np.sum(loss,1)
        return np.average(loss)

    def average_squared_error(self, test_points):
        """
        Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.  

        :param test_point: array, shape(visible_dim)
                           data point to test the reconstruction
        :return: sqr: float
                      error
        """
        ase_list=[]
        reconstruction, prob, _ = self.parallel_sample(test_points)
        as_e = tf.pow(test_points - reconstruction, 2)
        sqr = tf.reduce_sum(as_e, 1) / self._v_dim
        return np.mean(sqr)

    def free_energy(self,test_point):
        """
        Compute the free energy of the RBM, it is useful to compute the pseudologlikelihood.
        F(v) = - log \sum_h e^{-E(v,h)} = -bv - \sum_i \log(1 + e^{c_i + W_i v}) where v= visible state, h = hidden state,
        b = visible_biases, c = hidden_biases, W_i = i-th column of the weights matrix
        :param test_point: array, shape(visible_dim)
                           random point sampled from the test set
        :return: scalar
        """
        bv = tf.tensordot(test_point, tf.transpose(self.visible_biases),1)
        wx_b = tf.tensordot(self.weights,test_point,1) + self.hidden_biases
        hidden_term = tf.reduce_sum(tf.math.log(1+tf.math.exp(wx_b)))
        return np.asarray(-hidden_term -bv)[0]

    def pseudo_log_likelihood(self, test_point):
        i = np.random.randint(0,self._v_dim,1)
        test_point_flip = test_point.copy()
        test_point_flip[i] = np.logical_not(test_point_flip[i])
        fe_test = self.free_energy(test_point)
        fe_flip = self.free_energy(test_point_flip)
        pseudo = self._v_dim * tf.math.log(tf.sigmoid(fe_flip-fe_test))

        return pseudo

    def KL_divergence(self, data, n_points, k_neigh, MC_steps=1):
        #todo: I should try with reconstructing point starting from other points
        rnd_test_points_idx = np.random.randint(low=0, high=data['x_test'].shape[0], size=n_points)
        rnd_test_points_idx_2 = np.random.randint(low=0, high=data['x_test'].shape[0], size=n_points)
        test_points = data['x_test'][rnd_test_points_idx, :]
        test_points_2 = data['x_test'][rnd_test_points_idx_2, :]
        #reconstruction = np.empty(test_points_2.shape)
        reconstruction = self.parallel_sample(inpt=test_points_2, n_step_MC=MC_steps)[0]
        nbrs_data = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
        nbrs_data.fit(test_points)
        nbrs_model = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs =-1)
        nbrs_model.fit(reconstruction)

        rho, _ = nbrs_data.kneighbors(test_points)
        nu, _ = nbrs_model.kneighbors(test_points)

        rho_inv, _ = nbrs_data.kneighbors(reconstruction)
        nu_inv, _ = nbrs_model.kneighbors(reconstruction)

        l = 0
        l_inv = 0
        # -2 is needed because in rho the first distance is always 0 and then with the point itself that we should not consider,
        #to effectively pick the k-th neigh w.r.t test points and reconstructions we have to take the k-th in rho and the k-th -1 in nu.
        for i in range(n_points):
            l += np.log(nu[i, k_neigh-2] / rho[i, k_neigh-1])
            l_inv += np.log(rho_inv[i, k_neigh-2] / nu_inv[i, k_neigh-1])
        DKL = self._v_dim / n_points * l + np.log(n_points / (n_points - 1))
        DKL_inv = self._v_dim / n_points * l_inv + np.log(n_points / (n_points - 1))
        return DKL, DKL_inv

    def prob_beta(self,beta,points):
        p_A = np.exp((1-beta)*np.inner(points,self.visible_biases))*np.product(1+np.exp((1-beta)*self.hidden_biases))
        p_B = np.exp(beta*np.inner(points,self.visible_biases))*np.product(1+np.exp(beta*tf.tensordot(points, self.weights, axes=[[1], [1]]) + self.hidden_biases))

        return p_A*p_B

    def log_partition_function(self,batch,n_step=1, n_beta = 1000):
        beta = np.linspace(0,1,n_beta)
        rate = np.ones((beta.shape[0]-1, batch.shape[0], self._h_dim))
        for k,b in enumerate(beta[:-1]):
            hidden_probabilities_0_A = tf.sigmoid((1-b)*self.hidden_biases)  # dimension W + 1 row for biases
            hidden_probabilities_0_B = tf.sigmoid(b*(tf.tensordot(batch, self.weights, axes=[[1], [1]]) + self.hidden_biases))  # dimension W + 1 row for biases
            hidden_states_0_A = self.calculate_state(hidden_probabilities_0_A)
            hidden_states_0_B = self.calculate_state(hidden_probabilities_0_B)

            for _ in range(n_step):  # gibbs update
                visible_probabilities_1_AB = tf.sigmoid((1-b)*self.visible_biases + b*(tf.tensordot(hidden_states_0_B, self.weights, axes=[[1], [0]]) + self.visible_biases)) # dimension W + 1 row for biases
                visible_states_1_AB = self.calculate_state(visible_probabilities_1_AB)

                hidden_probabilities_1_A = tf.sigmoid((1 - b) * self.hidden_biases)  # dimension W + 1 row for biases
                hidden_probabilities_1_B = tf.sigmoid(b * (tf.tensordot(visible_states_1_AB, self.weights, axes=[[1], [1]]) + self.hidden_biases))  # dimension W + 1 row for biases
                hidden_states_1_A = self.calculate_state(hidden_probabilities_1_A)
                hidden_states_1_B = self.calculate_state(hidden_probabilities_1_B)

                hidden_states_0_A = hidden_states_1_A
                hidden_states_0_B = hidden_states_1_B
            p_k = ((1 + np.exp((1 - b) * self.hidden_biases)) * (1 + np.exp(b * tf.tensordot(visible_states_1_AB, self.weights, axes=[[1], [1]]) + self.hidden_biases))).astype(np.float64)
            p_k_1 = ((1 + np.exp((1 - beta[k+1]) * self.hidden_biases)) * (1 + np.exp(beta[k+1] * tf.tensordot(visible_states_1_AB, self.weights, axes=[[1], [1]]) + self.hidden_biases))).astype(np.float64) #p_(k)
            rate_b = (p_k_1 / p_k).astype(np.float64)
            rate[k] = rate_b
        w = np.product(rate, 0)
        logr_ais = np.sum(np.log(np.mean(w,0)))
        logZ_A = np.sum(np.log(1+tf.exp(self.visible_biases))) + np.sum(np.log(1+tf.exp(self.hidden_biases))) #if needed add astype(np.float64)

        return -logr_ais+logZ_A

    def log_likelihood(self,points,test):
        """

        :param points: data points to calculate likelihood
        :param test: some random points to start the gibbs sampling to estimate the partition function
        :return: float
        """
        # probably I should calculate the partition function just once
        log_partition_function = self.log_partition_function(test)
        #a lot of dubts wheter I should calculate the partition function on test or train and if i should calculate the likelihood for all the points or not
        log_L = np.inner(self.visible_biases.numpy(),points) + np.sum(np.log((1+np.exp(tf.tensordot(points.astype(np.float64), self.weights.numpy().astype(np.float64), axes=[[1], [1]]) + self.hidden_biases.numpy().astype(np.float64)))),1)

        return - np.mean(log_L) + log_partition_function/points.shape[0]

    def variable_summaries(self,var, step):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, step)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, step)
            tf.summary.scalar('max', tf.reduce_max(var), step)
            tf.summary.scalar('min', tf.reduce_min(var), step)
            tf.summary.histogram('histogram', var, step = step)

    def train(self, data, optimizer):
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
        test_fixed = np.random.randint(low=0, high=data['x_test'].shape[0], size=self.n_test_samples)
        for epoch in range(1,self._n_epoch+1):
            self.epoch = epoch
            #sys.stdout.write('\r')
            print('Epoch:',epoch, '/', self._n_epoch)
            np.random.shuffle(data['x_train'])
            #with tf.name_scope('Learning rate'):
                #learning_rate = self.exp_decay_l_r(epoch)
            for i in tqdm(range(0, data['x_train'].shape[0], self._batch_size)):


                '''
                x_train_mini = data['x_train'][i:i+self._batch_size]
                batch_dw = np.zeros((self._h_dim, self._v_dim, self._batch_size)) #d_w,d_v,d_h don't know why but this is not working
                batch_dvb = np.zeros((self._v_dim, self._batch_size))
                batch_dhb = np.zeros((self._h_dim, self._batch_size))

                # I should create an optimizer class at the moment is just if

                if self.training_algorithm == 'cd':
                    for ind,vec in enumerate(x_train_mini):
                        batch_dw[:,:,ind],batch_dvb[:,ind],batch_dhb[:,ind] = self.contr_divergence(vec, L2_l=0) #d_w,d_v,d_h not working get lost to write down the values

                #Persistent contrastive divergence
                elif self.training_algorithm == 'pcd':
                    start_point = x_train_mini[np.random.randint(0,self._batch_size,1)].reshape(self._v_dim)
                    for ind in range(self._batch_size):
                        batch_dw[:, :, ind], batch_dvb[:, ind], batch_dhb[:, ind], last_state = self.contr_divergence(start_point) #I removed last_state from output of contrastive divergence
                        start_point = tf.reshape(last_state,(self._v_dim,))

                self.grad_dict = {'weights': np.average(batch_dw,2),
                                   'visible_biases': np.average(batch_dvb,1),
                                   'hidden_biases': np.average(batch_dhb,1)}

                '''
                if self.training_algorithm == 'cd':
                    x_train_mini = data['x_train'][i:i + self._batch_size]
                    batch_dw, batch_dvb, batch_dhb= self.parallel_cd(x_train_mini)


                elif self.training_algorithm == 'pcd':
                    start_point = data['x_train'][i].reshape(self._v_dim)
                    x_train_mini = np.array(self.sample(start_point,self._batch_size)[3])
                    batch_dw, batch_dvb, batch_dhb = self.parallel_cd(x_train_mini)

                self.grad_dict = {'weights': batch_dw,
                                  'visible_biases': batch_dvb,
                                  'hidden_biases': batch_dhb}
                optimizer.fit()
            #Save model every epoch
            self.save_model()

            #test every epoch
            np.random.shuffle(data['x_test'])
            rnd_test_points_idx = np.random.randint(low = 0,high = data['x_test'].shape[0], size=self.n_test_samples) #sample size random points indexes from test
            with tf.name_scope('Errors'): #TODO: I should computer the reconstruction once and use it inside all these estimatiojs
                rec_error = self.reconstruction_cross_entropy(data['x_test'][rnd_test_points_idx,:]) #TODO: add random test datapoint
                sq_error = self.average_squared_error(data['x_test'][rnd_test_points_idx,:])
                free_energy = self.free_energy(data['x_test'][rnd_test_points_idx[0],:])
                pseudo_log = self.pseudo_log_likelihood(data['x_test'][rnd_test_points_idx[0],:])
                recon_c_e = self.recon_c_e(data['x_test'][rnd_test_points_idx,:])
                DKL, DKL_inv = self.KL_divergence(data,1000,7)
                log_L = self.log_likelihood(data['x_test'],data['x_test'][rnd_test_points_idx,:])
                tf.summary.scalar('rec_error', rec_error, step = epoch)
                tf.summary.scalar('squared_error', sq_error, step = epoch)
                tf.summary.scalar('Free Energy', free_energy, step = epoch)
                tf.summary.scalar('Pseudo log likelihood', pseudo_log, step=epoch)
                tf.summary.scalar('Binary cross entropy', recon_c_e, step=epoch)
                tf.summary.scalar('KL divergence', DKL, step=epoch)
                tf.summary.scalar('inverse KL divergence', DKL_inv, step=epoch)
                tf.summary.scalar('Log Likelihood', log_L, step=epoch)
            with tf.name_scope('Weights'):
                self.variable_summaries(self.weights, step = epoch)
            with tf.name_scope('hidden_biases'):
                self.variable_summaries(self.hidden_biases, step = epoch)
            with tf.name_scope('visible_biases'):
                self.variable_summaries(self.visible_biases, step=epoch)

            reconstruction_plot,prob,inpt,_ = self.sample(inpt=data['x_test'][rnd_test_points_idx[0],:])
            pic = tf.concat([tf.reshape(inpt,(1,self._v_dim)),prob,reconstruction_plot],0)
            tf.summary.image('Reconstruction pictures ',tf.reshape(pic,(3,self._picture_shape[0],self._picture_shape[1],1)),max_outputs=100,step = epoch)

            print("epoch %d" % (epoch + 1),"Rec error: %s" % np.asarray(rec_error),"sq_error %s" % np.asarray(sq_error))
