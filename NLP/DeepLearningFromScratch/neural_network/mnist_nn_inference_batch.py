import sys
import os
sys.path.append(os.pardir)


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


def calculate_batch_accuracy():
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i+batch_size]
        y = predict(network, x_batch)
        # axis=0表示列，1表示行
        p = np.argmax(y, axis=1)
        accuracy_cnt += np.sum(p == t[i: i+batch_size])

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))


def test_np_sum():
    y = np.array([1, 2, 3, 4])
    t = np.array([1, 1, 2, 4])

    print(t == y)
    print(np.sum(t == y))


if __name__ == '__main__':
    test_np_sum()
    calculate_batch_accuracy()
