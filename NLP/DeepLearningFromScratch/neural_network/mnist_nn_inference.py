import sys
import os
sys.path.append(os.pardir)
print(os.pardir)

from dataset.mnist import load_mnist
from softmax import softmax
import numpy as np
import pickle


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def calculate_accuracy():
    network = init_network()
    x, t = get_data()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        # 获取的是最大值的索引
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print('Accuracy:' + str(float(accuracy_cnt) / len(x)))


if __name__ == '__main__':
    calculate_accuracy()