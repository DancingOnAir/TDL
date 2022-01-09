import os
import sys
sys.path.append(os.pardir)


from common.layers import Sigmoid, Relu, SoftmaxWithLoss, Affine
from common.functions import cross_entropy_error
from collections import OrderedDict
import numpy as np


class MulLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_decay_lambda = weight_decay_lambda
        self.params = dict()

        # 初始化权重
        self.__init_weight(weight_init_std)
        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for i in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            # 注意最后的语法activation_layer[activation]()
            self.layers['Activation_function' + str(i)] = activation_layer[activation]()

        i = self.hidden_layer_num + 1
        self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[i - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[i - 1])

            self.params['W' + str(i)] = scale * np.random.randn(all_size_list[i - 1], all_size_list[i])
            self.params['b' + str(i)] = np.zeros(all_size_list[i])

    def predict(self, x):
        pass

    def loss(self, x, t):
        pass

    def accuracy(self, x, t):
        pass


