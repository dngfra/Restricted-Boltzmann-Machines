import tensorflow as tf 
import numpy as np 
import datetime
import math
import sys
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
'''
class monitoring():
    def reconstruction_cross_e(): 
    
    def ava_sq_error(): 
    
    def pseudo_log_l():
'''

class RBM():
    def __init__(self, visible_dim, hidden_dim,  number_of_epochs, batch_size, init_learning_rate = 0.01):
        self._n_epoch = number_of_epochs
        self._v_dim = visible_dim
        self._h_dim = hidden_dim
        self._l_r = init_learning_rate
        self._batch_size = batch_size
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
    #@tf.function
    def calculate_state(self, probability):
        binom = tf.concat([1-probability,probability],0)
        prob_re = tf.reshape(binom,(2,probability.get_shape()[1]))
        #print("check summation probability:", tf.reduce_sum(prob_re,0)) # check prob summation to 1
        return tf.reshape(tf.cast(tf.random.categorical(tf.math.log(tf.transpose(prob_re)),1), tf.float32), (1,probability.get_shape()[1]))
   

    #@tf.function
    def sample(self, inpt = None ,n_step_MC=1): #the error is probably because i cannot use self here inside
        if inpt.any() == None:
            inpt = tf.constant(np.random.randint(2, size=self._v_dim))
        hidden_probabilities_0 = tf.sigmoid(tf.add(tf.tensordot(self.weights, inpt,1), self.hidden_biases)) # dimension W + 1 row for biases
        hidden_states_0 = self.calculate_state(hidden_probabilities_0)
        for _ in range(n_step_MC): #gibbs update
            visible_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(hidden_states_0,self.weights,1), self.visible_biases)) # dimension W + 1 row for biases
            visible_states_1 = self.calculate_state(visible_probabilities_1)
            hidden_probabilities_1 = tf.sigmoid(tf.add(tf.tensordot(visible_states_1, tf.transpose(self.weights),1), self.hidden_biases)) # dimension W + 1 row for biases
            hidden_states_1 = self.calculate_state(hidden_probabilities_1)
            hidden_states_0 = hidden_states_1
        return visible_states_1,visible_probabilities_1

    #@tf.function
    def contr_divergence(self, data_point, n_step_MC=1): #TODO: add regularization term, I could use sample in the following
        """
        Perform contrastive divergence given a data point.
        :param data_point: array, shape(visible layer)
                           data point sampled from the batch
        :param n_step_MC: int
                          tep of the markov chain for the sampling (CD1,CD2,...)
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

        vh_0 = tf.reshape(tf.tensordot(hidden_states_0_copy, data_point, 0), (200,784))
        vh_1 = tf.reshape(tf.tensordot(hidden_states_1, visible_states_1, 0), (200,784))
        delta_w = tf.add(vh_0, - vh_1)
        delta_vb = tf.add(data_point, - visible_states_1)
        delta_hb = tf.add(hidden_states_0_copy, - hidden_states_1)
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



    def reconstruction_cross_entropy(self,test_point, plot=True):
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
        reconstruction,prob = self.sample(inpt=test_point)
        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(test_point.reshape(28, 28),cmap='Greys')
            axes[0].set_title("Original Image")
            axes[1].imshow(np.asarray(reconstruction).reshape(28, 28), cmap='Greys')
            axes[1].set_title("Reconstruction")
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        #tf.where is needed to have 0*-\infty = 0
        r_ce = tf.multiply(reconstruction, tf.where(tf.math.is_inf(tf.math.log(prob)),np.zeros_like(tf.math.log(prob)),tf.math.log(prob))) \
               + tf.multiply((1-reconstruction), tf.where(tf.math.is_inf(tf.math.log(1-prob)),np.zeros_like(tf.math.log(1-prob)), tf.math.log(1-prob)))

        return -tf.reduce_sum(r_ce,1)[0] # TODO: check if this axis is actually correct

    def average_squared_error(self, test_point):
        """
        Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.  
        :param test_point: array, shape(visible_dim)
                           data point to test the reconstruction
        :return: sqr: float
                      error
        """
        reconstruction,_ = self.sample(inpt = test_point)
        as_e = tf.pow(test_point - reconstruction,2)
        sqr = tf.reduce_sum(as_e,1)/self._v_dim
        return sqr[0] #TODO: check if the axis for the sum is correct

    def free_energy(self,test_point):
        bv = tf.tensordot(test_point, tf.transpose(self.visible_biases),1)
        wx_b = tf.tensordot(self.weights,test_point,1) + self.hidden_biases
        hidden_term = tf.reduce_sum(tf.math.log(1+tf.math.exp(wx_b)))
        return np.asarray(-hidden_term -bv)[0]

    def pseudo_log_likelihood(self):

        return self

    def KL_divergence(self):

        return self

    def exp_decay_l_r(self,epoch):
        initial_lrate = self._l_r
        k = 0.1
        lrate = initial_lrate * np.exp(-k * epoch)
        return lrate


    def train(self, data):
        """
        This function shuffle the dataset and create #data_train/batch_size mini batches and perform contrastive divergence on each vector of the batch. 
        The upgrade of the parameters is performed only at the end of each batch by taking the average of the gradients on the batch. 
        In the last part a random datapoint is sampled from the test set to calculate the error reconstruction. The entire procedure is repeted 
         _n_epochs times.
        :param data: dict
                     dictionary of numpy arrays with ['x_train','y_train','x_test', 'y_test']
        :return: self
        """
        for epoch in range(self._n_epoch):
            sys.stdout.write('\r')
            np.random.shuffle(data['x_train'])
            #learning_rate = self.exp_decay_l_r(epoch)
            for i in tqdm(range(0, data['x_train'].shape[0], self._batch_size)):
                x_train_mini = data['x_train'][i:i+self._batch_size]
                batch_dw = np.zeros((self._h_dim, self._v_dim, self._batch_size)) #d_w,d_v,d_h don't know why but this is not working
                batch_dvb = np.zeros((self._v_dim, self._batch_size))
                batch_dhb = np.zeros((self._h_dim, self._batch_size))
                for ind,vec in enumerate(x_train_mini):
                    #print(ind)
                    batch_dw[:,:,ind],batch_dvb[:,ind],batch_dhb[:,ind] = self.contr_divergence(vec) #d_w,d_v,d_h not working get lost to write down the values

                dw = np.average(batch_dw,2)
                dvb = np.average(batch_dvb,1)
                dhb = np.average(batch_dhb,1)
                self.weights = self.weights + self._l_r * dw
                self.visible_biases = self.visible_biases + self._l_r* dvb
                self.hidden_biases = self.hidden_biases + self._l_r* dhb
            #test every epoch
            random_test_point = random.randint(0,data['x_test'].shape[0]-1) #read documentation for -1
            with tf.name_scope('rec_error'): #TODO: fix the sample and not the point
                rec_error = self.reconstruction_cross_entropy(data['x_test'][random_test_point]) #TODO: add random test datapoint
            tf.summary.scalar('rec_error', rec_error, step = epoch)
            with tf.name_scope('squared_error'):
                sq_error = self.average_squared_error(data['x_test'][random_test_point])
            tf.summary.scalar('squared_error', sq_error, step = epoch)
            with tf.name_scope('Free Energy'):
                free_energy = self.free_energy(data['x_test'][random_test_point])
            tf.summary.scalar('Free Energy', free_energy, step = epoch)

            print("epoch %d" % (epoch + 1),"Rec error: %s" % np.asarray(rec_error),"sq_error %s" % np.asarray(sq_error))