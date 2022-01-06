import os
import sys
sys.path.append(os.pardir)


from common.layers import *
from common.gradient import numerical_gradient
import numpy as np


class MulLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu', weight_decay_lambda=0, use_dropout=False, dropout_rate=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = dict()

        self.__init_weight(weight_init_std)

        self.last_layer = SoftmaxWithLoss()


    # xavier and he
    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(all_size_list)):
            scale = weight_init_std
            if scale.lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[i - 1])
            elif scale.lower() in ('sigmoid', 'tanh'):
                scale = 1.0 / np.sqrt(all_size_list[i - 1])

            self.params['W' + str(i)] = np.random.randn(all_size_list[i - 1], all_size_list[i])
            self.params['b' + str(i)] = np.zeros_like(all_size_list[i])

    def predict(self, x):
        pass

    def loss(self, x, t, train_flag=False):


    def accuracy(self, X, T):
        pass




