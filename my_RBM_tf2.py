import tensorflow as tf 
import numpy as np 
import datetime
import math
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import h5py
import deepdish as dd

'''
class monitoring():
    def reconstruction_cross_e(): 
    
    def ava_sq_error(): 
    
    def pseudo_log_l():
'''

class RBM():
    def __init__(self, visible_dim, hidden_dim,  number_of_epochs, batch_size,n_test_samples=100, init_learning_rate = 0.8):
        self._n_epoch = number_of_epochs
        self._v_dim = visible_dim
        self._h_dim = hidden_dim
        self._l_r = init_learning_rate
        self._batch_size = batch_size
        self.n_test_samples = n_test_samples
        self._current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._log_dir = 'logs/scalars/' + self._current_time + '/train'
        self._file_writer = tf.summary.create_file_writer(self._log_dir)
        self._file_writer.set_as_default()
        #self.model = self.model
        self.visible_biases = tf.Variable(tf.random.uniform([1, visible_dim], 0, 0.1), tf.float32, name="visible_biases")
        self.hidden_biases = tf.Variable(tf.random.uniform([1, hidden_dim], 0, 0.1), tf.float32, name="hidden_biases")
        self.weights = tf.Variable(tf.random.normal([hidden_dim, visible_dim], mean = 0.0, stddev = 0.1), tf.float32, name="weights")

    '''
    @tf.function    
    def model(self):
        self.visible_biases = tf.Variable(tf.random.uniform([1, self._v_dim], 0, 1), tf.float32, name="visible_biases")
        self.hidden_biases = tf.Variable(tf.random.uniform([1, self._h_dim], 0, 1), tf.float32, name="hidden_biases")
        self.weights = tf.Variable(tf.random.normal([self._h_dim,self._v_dim], mean = 0.0, stddev = 0.01),tf.float32, name="weights")
        #self.learning_rate = tf.Variable(tf.fill([self._v_dim, self._h_dim], learning_rate), name="learning_rate")
    '''
    def save_model(self):
        """
        Save the current RBM model as .h5 file dictionary with  keys: {'weights', 'visible biases', 'hidden_biases' }
        """
        model_dict = {'weights': np.asarray(self.weights), 'visible biases': np.asarray(self.visible_biases), 'hidden biases': np.asarray(self.hidden_biases)}
        return dd.io.save('results/models/'+self._current_time+'model.h5', model_dict)

    def from_saved_model(self,model_path):
        """

        :param model_path: string
                           path of .h5 file containing dictionary of the model with  keys: {'weights', 'visible biases', 'hidden biases' }
        :return: loaded model
        """
        model_dict = dd.io.load(model_path)
        self.weights = model_dict['weights']
        self.visible_biases = model_dict['visible biases']
        self.hidden_biases = model_dict['hidden biases']
        return self

    #@tf.function
    def calculate_state(self, probability):
        """
        Given the probability(x'=1|W,b) = sigmoid(Wx+b) computes the next state by sampling from the relative binomial distribution.
        x and x' can be the visible and hidden layers respectively or viceversa.
        :param probability: array, shape(visible_dim) or shape(hidden_dim)
        :return: array , shape(visible_dim) or shape(hidden_dim)
                 0,1 state of each unit in the layer
        """
        binom = tf.concat([1-probability,probability],0)
        prob_re = tf.reshape(binom,(2,probability.get_shape()[1]))
        #print("check summation probability:", tf.reduce_sum(prob_re,0)) # check prob summation to 1
        return tf.reshape(tf.cast(tf.random.categorical(tf.math.log(tf.transpose(prob_re)),1), tf.float32), (1,probability.get_shape()[1]))
   

    #@tf.function
    def sample(self, inpt = [] ,n_step_MC=1,p_0=0.5,p_1=0.5): #TODO: add p_0 and p_1 in the arguments
        """
        Sample from the RBM with n_step_MC steps markov chain.
        :param inpt: array shape(visible_dim)
                     It is possible to start the markov chain from a given point from the dataset or from a random state
        :param n_step_MC: scalar
                          number of markov chain steps to sample.
        :return: visible_states_1: array shape(visible_dim)
                 visible state after n_step_MC steps
                 visible_probabilities_1: array shape(visible_dim)
                 probabilities from which visible_states_1 is sampled
        """
        if len(inpt) == 0:
            #inpt = tf.constant(np.random.randint(2, size=self._v_dim), tf.float32)
            inpt = tf.constant(np.random.choice([0,1], size=self._v_dim,p=[p_0,p_1]), tf.float32)
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, inpt,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        for _ in range(n_step_MC): #gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0,self.weights,1), self.visible_biases)) # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
        return visible_states_1,visible_probabilities_1,inpt

    #@tf.function
    def contr_divergence(self, data_point, n_step_MC=1, L2_l = 0): #TODO: I could use sample in the following
        """
        Perform contrastive divergence given a data point.
        :param data_point: array, shape(visible layer)
                           data point sampled from the batch
        :param n_step_MC: int
                          tep of the markov chain for the sampling (CD1,CD2,...)
        :param L2_l: float, lambda for L2 regularization, default = 0 so no regularization performed
        :return: delta_w: array shape(hidden_dim, visible_dim)
                          Array of the same shape of the weight matrix which entries are the gradients dw_{ij}
                 delta_vb: array, shape(visible_dim)
                           Array of the same shape of the visible biases which entries are the gradients d_vb_i
                 delta_vb: array, shape(hidden_dim)
                           Array of the same shape of the hidden biases which entries are the gradients d_hb_i

        """
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, data_point,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        hidden_states_0_copy = hidden_states_0
        for _ in range(n_step_MC): #gibbs update
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
        return delta_w, delta_vb, delta_hb


    #@tf.function
    def persistent_contr_divergence(self, data):
        """
        Persistent CD [Tieleman08] uses another approximation for sampling from p(v,h).
        It relies on a single Markov chain, which has a persistent state (i.e., not restarting
        a chain for each observed example). For each parameter update, we extract new samples by
        simply running the chain for k-steps. The state of the chain is then preserved for subsequent updates.
        """
        return



    def reconstruction_cross_entropy(self,test_points, plot=True):
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
        #TODO: do we need to average over multiple test point?
        
        r_ce_list=[]
        for vec in test_points: 
            reconstruction,prob,_ = self.sample(inpt=vec)
            #tf.where is needed to have 0*-\infty = 0
            r_ce = tf.multiply(reconstruction, tf.where(tf.math.is_inf(tf.math.log(prob)),np.zeros_like(tf.math.log(prob)),tf.math.log(prob))) \
                   + tf.multiply((1-reconstruction), tf.where(tf.math.is_inf(tf.math.log(1-prob)),np.zeros_like(tf.math.log(1-prob)), tf.math.log(1-prob)))
            r_ce_list.append(-tf.reduce_sum(r_ce,1)[0])

        if plot:
            reconstruction_plot, _, _= self.sample(inpt=test_points[1,:])
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(test_points[1,:].reshape(28, 28),cmap='Greys')
            axes[0].set_title("Original Image")
            axes[1].imshow(np.asarray(reconstruction_plot).reshape(28, 28), cmap='Greys')
            axes[1].set_title("Reconstruction")
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        return np.average(r_ce_list)

    def average_squared_error(self, test_points):
        """
        Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.  
        :param test_point: array, shape(visible_dim)
                           data point to test the reconstruction
        :return: sqr: float
                      error
        """
        ase_list=[]
        for vec in test_points:
            reconstruction,_,_= self.sample(inpt = vec)
            as_e = tf.pow(vec - reconstruction,2)
            sqr = tf.reduce_sum(as_e,1)/self._v_dim
            ase_list.append(sqr[0])
            return np.mean(ase_list)

    def free_energy(self,test_point):
        """
        Compute the free energy of the RBM, it is useful to compute the pseudologlikelihood.
        F(v) = - log \sum_h e^{-E(v,h)} = -bv - \sum_i \log(1 + e^{c_i + W_i v}) where v= visible state, h = hidden state,
        b = visible biases, c = hidden biases, W_i = i-th column of the weights matrix
        :param test_point: array, shape(visible_dim)
                           random point sampled from the test set
        :return: scalar
        """
        bv = tf.tensordot(test_point, tf.transpose(self.visible_biases),1)
        wx_b = tf.tensordot(self.weights,test_point,1) + self.hidden_biases
        hidden_term = tf.reduce_sum(tf.math.log(1+tf.math.exp(wx_b)))
        return np.asarray(-hidden_term -bv)[0]

    def pseudo_log_likelihood(self):

        return self

    def KL_divergence(self):

        return self

    def exp_decay_l_r(self,epoch):
        """
        When training a model, it is often recommended to lower the learning rate as the training progresses.
        This function applies an exponential decay function to a provided initial learning rate.
        :param epoch: scalar
        :return: scalar
        """
        k = 0.1
        lrate = self._l_r * np.exp(-k * epoch)
        return lrate

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

    def train(self, data):
        """
        This function shuffle the dataset and create #data_train/batch_size mini batches and perform contrastive divergence on each vector of the batch. 
        The upgrade of the parameters is performed only at the end of each batch by taking the average of the gradients on the batch. 
        In the last part a random datapoint is sampled from the test set to calculate the error reconstruction. The entire procedure is repeted 
         _n_epochs times.
        :param data: dict
                     dictionary of numpy arrays with labels ['x_train','y_train','x_test', 'y_test']
        :return: self
        """
        print('Start training...')
        for epoch in range(self._n_epoch):
            sys.stdout.write('\r')
            np.random.shuffle(data['x_train'])
            with tf.name_scope('Learning rate'):
                learning_rate = self.exp_decay_l_r(epoch)
            for i in tqdm(range(0, data['x_train'].shape[0], self._batch_size)):
                x_train_mini = data['x_train'][i:i+self._batch_size]
                batch_dw = np.zeros((self._h_dim, self._v_dim, self._batch_size)) #d_w,d_v,d_h don't know why but this is not working
                batch_dvb = np.zeros((self._v_dim, self._batch_size))
                batch_dhb = np.zeros((self._h_dim, self._batch_size))
                for ind,vec in enumerate(x_train_mini):
                    #print(ind)
                    batch_dw[:,:,ind],batch_dvb[:,ind],batch_dhb[:,ind] = self.contr_divergence(vec, L2_l=0) #d_w,d_v,d_h not working get lost to write down the values

                dw = np.average(batch_dw,2)
                dvb = np.average(batch_dvb,1)
                dhb = np.average(batch_dhb,1)
                self.weights = self.weights + learning_rate * dw
                self.visible_biases = self.visible_biases + learning_rate* dvb
                self.hidden_biases = self.hidden_biases + learning_rate* dhb
            #Save model every epoch
            self.save_model()

            #test every epoch
            np.random.shuffle(data['x_test'])
            rnd_test_points_idx = np.random.randint(low = 0,high = data['x_test'].shape[0], size=self.n_test_samples) #sample size random points indexes from test
            with tf.name_scope('Errors'): #TODO: fix the sample and not the point
                rec_error = self.reconstruction_cross_entropy(data['x_test'][rnd_test_points_idx,:]) #TODO: add random test datapoint
                sq_error = self.average_squared_error(data['x_test'][rnd_test_points_idx,:])
                free_energy = self.free_energy(data['x_test'][rnd_test_points_idx[0],:])
            tf.summary.scalar('rec_error', rec_error, step = epoch)
            tf.summary.scalar('squared_error', sq_error, step = epoch)
            tf.summary.scalar('Free Energy', free_energy, step = epoch)
            tf.summary.scalar('Learning rate', learning_rate, step = epoch)
            with tf.name_scope('Weights'):
                self.variable_summaries(self.weights, step = epoch)
            with tf.name_scope('Hidden biases'):
                self.variable_summaries(self.hidden_biases, step = epoch)
            with tf.name_scope('Visible biases'):
                self.variable_summaries(self.visible_biases, step=epoch)

            print("epoch %d" % (epoch + 1),"Rec error: %s" % np.asarray(rec_error),"sq_error %s" % np.asarray(sq_error))