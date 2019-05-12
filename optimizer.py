import numpy as np
import tensorflow as tf
class Optimizer:
    def __init__(self, machine, learning_rate, eps = 1e-8, opt = 'mini_SGD'):
        self.machine = machine
        self.learning_rate = learning_rate
        self.eps = eps
        self.opt = opt


    def adam(self):
        """
        Update the model using adam optimizer.
        :param model: Dictionary of weights and biases i.e. model = dict(W=..., vb=..., hb=...)
        :param grad: Dictionary of gradients i.e. model = dict(W=..., vb=..., hb=...) #same keys
        :param epoch: int
        :return:
        """
        self.M = {k: np.zeros_like(v).astype(np.float32) for k, v in self.machine.model_dict.items()}
        self.R = {k: np.zeros_like(v).astype(np.float32) for k, v in self.machine.model_dict.items()}
        beta1 = .9
        beta2 = .999
        for key, value in self.machine.grad_dict.items(): #iteration on keys
            self.M[key] = beta1 * self.M[key] + (1. - beta1) * value
            self.R[key] = beta2 * self.R[key] + (1. - beta2) * value ** 2

            m_k_hat = self.M[key] / (1. - beta1 ** (self.machine.epoch))
            r_k_hat = self.R[key] / (1. - beta2 ** (self.machine.epoch))

            self.machine.model_dict[key].assign_add((self.learning_rate * m_k_hat / (np.sqrt(r_k_hat) + self.eps)).reshape(self.machine.model_dict[key].shape))
        return self.machine.update_model()

    def exp_decay_l_r(self,k=0.1):
        """
        When training a model, it is often recommended to lower the learning rate as the training progresses.
        This function applies an exponential decay function to a provided initial learning rate.
        :param epoch: scalar
        :return: scalar
        """
        lrate = self.learning_rate * np.exp(-k * self.machine.epoch)
        return lrate

    def minibatch_SGD(self):
        l_r = self.exp_decay_l_r()
        for key, value in self.machine.model_dict.items():
            self.machine.model_dict[key] = value + self.machine.grad_dict[key]*l_r
        return self.machine.update_model()


    def fit(self):
        if self.opt == 'mini_SGD':
            self.minibatch_SGD()
        elif self.opt == 'adam':
            self.adam()