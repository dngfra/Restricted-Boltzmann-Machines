import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class Metrics_monitor:
    def __init__(self, machine, metrics = ['rec_error','sq_error','pseudo_log','DKL','recon_c_e','recon_c_e','sparsity','magn_error','magn_error','log_like_AIS'] , steps = None):
        self.machine = machine
        self.steps = steps
        self.metrics = metrics

    def reconstruction_cross_entropy(self,samples, prob):
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

        reconstruction = samples
        # tf.where is needed to have 0*-\infty = 0
        r_ce = tf.multiply(reconstruction,
                           tf.where(tf.math.is_inf(tf.math.log(prob)), np.zeros_like(tf.math.log(prob)),
                                    tf.math.log(prob))) \
               + tf.multiply((1 - reconstruction),
                             tf.where(tf.math.is_inf(tf.math.log(1 - prob)), np.zeros_like(tf.math.log(1 - prob)),
                                      tf.math.log(1 - prob)))

        return tf.reduce_mean(r_ce)

    def recon_c_e(self, data,samples_prob):
        loss = tf.keras.backend.binary_crossentropy(data, samples_prob).numpy()
        loss = np.sum(loss, 1)
        return np.average(loss)

    def sparsity(self,data):
        hidden = self.machine.forward(data)
        return tf.reduce_mean(hidden)


    def average_squared_error(self, data,samples):
        """
        Compute the mean squared error between a test vector and its reconstruction performed by the RBM, ||x - z||^2.

        :param test_point: array, shape(visible_dim)
                           data point to test the reconstruction
        :return: sqr: float
                      error
        """
        as_e = tf.pow(data - samples, 2)
        sqr = tf.reduce_sum(as_e, 1) / self.machine._v_dim
        return np.mean(sqr)


    def pseudo_log_likelihood(self, data):
        i = np.random.randint(0, self.machine._v_dim, data.shape[0])
        j = np.arange(0, data.shape[0])
        b = np.zeros_like(data)
        b[j, i] = 1
        test_point_flip = np.abs(data - b)
        fe_test = self.machine.clamped_free_energy(data,False)
        fe_flip = self.machine.clamped_free_energy(test_point_flip,False)
        pseudo = tf.reduce_mean(self.machine._v_dim * tf.math.log(tf.sigmoid(fe_flip - fe_test)))

        return pseudo

    def KL_divergence(self, data,samples, k_neigh):
        # todo: I should try with reconstructing point starting from other points toavoid biases
        n_points = data.shape[0]
        test_points = data
        nbrs_data = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs=-1)
        nbrs_data.fit(test_points)
        nbrs_model = NearestNeighbors(n_neighbors=k_neigh, algorithm='ball_tree', metric='jaccard', n_jobs=-1)
        nbrs_model.fit(samples)

        rho, _ = nbrs_data.kneighbors(test_points)
        nu, _ = nbrs_model.kneighbors(test_points)

        rho_inv, _ = nbrs_data.kneighbors(samples)
        nu_inv, _ = nbrs_model.kneighbors(samples)

        l = 0
        l_inv = 0
        # -2 is needed because in rho the first distance is always 0 and then with the point itself.machine that we should not consider,
        # to effectively pick the k-th neigh w.r.t test points and reconstructions we have to take the k-th in rho and the k-th -1 in nu.
        for i in range(n_points):
            l += np.log(nu[i, k_neigh - 2] / rho[i, k_neigh - 1])
            l_inv += np.log(rho_inv[i, k_neigh - 2] / nu_inv[i, k_neigh - 1])
        DKL = self.machine._v_dim / n_points * l + np.log(n_points / (n_points - 1))
        DKL_inv = self.machine._v_dim / n_points * l_inv + np.log(n_points / (n_points - 1))
        return DKL, DKL_inv

    def AIS(self, n_beta=10000, n_conf=20):
        # configurations = np.random.choice([0, 1], size=(100,784), p=[0.5, 0.5])
        n_step = 1
        # standard versione without manipulation on expectation of ratio
        beta = np.linspace(0, 1, n_beta)
        batch = np.random.choice([0, 1], size=(n_conf, self.machine._v_dim), p=[0.5, 0.5]).astype(np.float64)
        # beta = np.concatenate([np.linspace(0,0.5,int(n_beta/4)),np.linspace(0.5,0.9,int(n_beta/4)),np.linspace(0.9,1,n_beta)])
        # rate = np.ones((beta.shape[0]-1, batch.shape[0], self.machine._h_dim))
        rate = np.ones((beta.shape[0] - 1, batch.shape[0], self.machine._h_dim))

        for k, b in enumerate(beta[:-1]):
            hidden_probabilities_0_A = tf.sigmoid((1 - b) * self.machine.hidden_biases)  # dimension W + 1 row for biases
            hidden_probabilities_0_B = tf.sigmoid(b * (tf.tensordot(batch, self.machine.weights, axes=[[1], [
                1]]) + self.machine.hidden_biases))  # dimension W + 1 row for biases
            hidden_states_0_A = self.machine.calculate_state(hidden_probabilities_0_A)
            hidden_states_0_B = self.machine.calculate_state(hidden_probabilities_0_B)

            for _ in range(n_step):  # gibbs update
                visible_probabilities_1_AB = tf.sigmoid((1 - b) * self.machine.visible_biases + b * (
                            tf.tensordot(hidden_states_0_B, self.machine.weights,
                                         axes=[[1], [0]]) + self.machine.visible_biases))  # dimension W + 1 row for biases
                visible_states_1_AB = self.machine.calculate_state(visible_probabilities_1_AB)

                hidden_probabilities_1_A = tf.sigmoid((1 - b) * self.machine.hidden_biases)  # dimension W + 1 row for biases
                hidden_probabilities_1_B = tf.sigmoid(b * (tf.tensordot(visible_states_1_AB, self.machine.weights, axes=[[1],[1]]) + self.machine.hidden_biases))  # dimension W + 1 row for biases
                hidden_states_1_A = self.machine.calculate_state(hidden_probabilities_1_A)
                hidden_states_1_B = self.machine.calculate_state(hidden_probabilities_1_B)

                hidden_states_0_A = hidden_states_1_A
                hidden_states_0_B = hidden_states_1_B
            batch = visible_states_1_AB
            if k == 0:
                p_k = 1 + tf.exp(self.machine.hidden_biases)
                p_k_1 = (1 + np.exp((1 - beta[k + 1]) * self.machine.hidden_biases)) * (1 + np.exp(beta[k + 1] * (
                            tf.tensordot(visible_states_1_AB, self.machine.weights,
                                         axes=[[1], [1]]) + self.machine.hidden_biases)))  # p_(k)
            elif k == n_beta - 2:
                p_k = (1 + np.exp((1 - b) * self.machine.hidden_biases)) * (1 + np.exp(
                    b * (tf.tensordot(visible_states_1_AB, self.machine.weights, axes=[[1], [1]]) + self.machine.hidden_biases)))
                p_k_1 = 1 + np.exp(beta[k + 1] * (tf.tensordot(visible_states_1_AB, self.machine.weights,
                                                               axes=[[1], [1]]) + self.machine.hidden_biases))
            else:
                p_k = (1 + np.exp((1 - b) * self.machine.hidden_biases)) * (1 + np.exp(
                    b * (tf.tensordot(visible_states_1_AB, self.machine.weights, axes=[[1], [1]]) + self.machine.hidden_biases)))
                p_k_1 = (1 + np.exp((1 - beta[k + 1]) * self.machine.hidden_biases)) * (1 + np.exp(beta[k + 1] * (
                            tf.tensordot(visible_states_1_AB, self.machine.weights,
                                         axes=[[1], [1]]) + self.machine.hidden_biases)))  # p_(k)
            rate[k, :, :] = (p_k_1 / p_k)
        rate = np.product(rate, 0)
        rate = np.mean(rate, 0)
        # w = np.product(rate)
        logr_ais = np.sum(np.log(rate))
        # logr_ais = np.log(np.mean(w,0))
        # logr_ais = np.log(w)
        # variance_w = 0 # np.std(w,0)
        logZ_A = np.sum(np.log(1 + tf.exp(self.machine.visible_biases))) + np.sum(
            np.log(1 + tf.exp(self.machine.hidden_biases)))  # if needed add astype(np.float64)
        return logZ_A + logr_ais

    def log_likelihood(self, points,  n_beta=20000, n_conf=20):
        """
        :param points: data points to calculate likelihood
        :param test: some random points to start the gibbs sampling to estimate the partition function
        :return: float
        """
        # probably I should calculate the partition function just once
        log_partition_function = self.AIS(n_beta, n_conf)
        # a lot of dubts wheter I should calculate the partition function on test or train and if i should calculate the likelihood for all the points or not
        log_L = -self.machine.clamped_free_energy(points)

        return log_L - log_partition_function, log_partition_function

    def magnetization_reconstruction(self, data,samples):
        magn_test = np.mean(data, 1)
        magn_reconstruction = np.mean(samples, 1)
        error = np.mean(np.abs((magn_test - magn_reconstruction) / magn_test))
        return error

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

    def fit(self,test_set, n_points = None):
        if self.steps == None:
            self.steps = self.machine.k

        if not n_points == None:
            rnd_test_points_idx = np.random.randint(low=0, high=test_set.shape[0],size=n_points)  # sample size random points indexes from test
            test_set = test_set[rnd_test_points_idx,:]
        samples, prob, _ = self.machine.parallel_sample(inpt=test_set, n_step_MC=self.steps)
        with tf.name_scope('Performance Metrics'):
            if 'rec_error' in self.metrics:
                rec_error = self.reconstruction_cross_entropy(samples,prob)
                tf.summary.scalar('rec_error', rec_error, step=self.machine.epoch)
            if 'sq_error' in self.metrics:
                sq_error = self.average_squared_error(test_set,samples)
                tf.summary.scalar('squared_error', sq_error, step=self.machine.epoch)
            if 'pseudo_log' in self.metrics:
                pseudo_log = self.pseudo_log_likelihood(test_set)
                tf.summary.scalar('Pseudo log likelihood', pseudo_log, step=self.machine.epoch)
            if 'DKL' in self.metrics:
                DKL, DKL_inv = self.KL_divergence( test_set, samples, 100)
                tf.summary.scalar('KL divergence', DKL, step=self.machine.epoch)
                tf.summary.scalar('inverse KL divergence', DKL_inv, step=self.machine.epoch)
            if 'recon_c_e' in self.metrics:
                recon_c_e = self.recon_c_e(test_set, prob)
                tf.summary.scalar('Binary cross entropy', recon_c_e, step=self.machine.epoch)
            if 'magn_error' in self.metrics:
                magnetization_reco_error = self.magnetization_reconstruction(test_set, samples)
                tf.summary.scalar('Magn reconstruction error', magnetization_reco_error, step=self.machine.epoch)
            if 'log_like_AIS' in self.metrics:
                log_L_AIS, logZ_AIS = self.log_likelihood(test_set)
                tf.summary.scalar('Log Likelihood AIS', log_L_AIS, step=self.machine.epoch)
            if 'sparsity' in self.metrics:
                spar = self.sparsity(test_set)
                tf.summary.scalar('Sparsity', spar, step=self.machine.epoch)
        with tf.name_scope('Weights'):
            self.variable_summaries(self.machine.weights, step=self.machine.epoch)
        with tf.name_scope('hidden_biases'):
            self.variable_summaries(self.machine.hidden_biases, step=self.machine.epoch)
        with tf.name_scope('visible_biases'):
            self.variable_summaries(self.machine.visible_biases, step=self.machine.epoch)
        with tf.name_scope('Gradients_weights'):
            self.variable_summaries(self.machine.grad_dict['weights'], step=self.machine.epoch)
        with tf.name_scope('Gradients_visible_biases'):
            self.variable_summaries(self.machine.grad_dict['visible_biases'], step=self.machine.epoch)
        with tf.name_scope('Gradients_hidden_biases'):
            self.variable_summaries(self.machine.grad_dict['hidden_biases'], step=self.machine.epoch)
        with tf.name_scope('Weights_norm:'):
            tf.summary.scalar('Weights_norm', np.linalg.norm(self.machine.weights.numpy()), step=self.machine.epoch)
        i = np.random.randint(low=0, high=test_set.shape[0], size=1)[0]
        reconstruction_plot, prob, inpt, _ = self.machine.sample(inpt=test_set[i,:])
        pic = tf.concat([tf.reshape(inpt, (1, self.machine._v_dim)), prob, reconstruction_plot], 0)
        tf.summary.image('Reconstruction pictures ',tf.reshape(pic, (3, self.machine._picture_shape[0], self.machine._picture_shape[1], 1)), max_outputs=100,
                         step=self.machine.epoch)
        print("epoch %d" % (self.machine.epoch + 1), "Rec error: %s" % np.asarray(recon_c_e),
              "sq_error %s" % np.asarray(sq_error))
