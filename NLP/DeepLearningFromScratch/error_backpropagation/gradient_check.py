import sys
import os
sys.path.append(os.pardir)

from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np


def test_gradient_check():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    for k in grad_numerical:
        diff = np.average(np.abs(grad_numerical[k] - grad_backprop[k]))
        print(k + ": " + str(diff))


if __name__ == '__main__':
    test_gradient_check()
