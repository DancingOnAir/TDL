import numpy as np


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    exp_x_sum = np.sum(exp_x)
    y = exp_x / exp_x_sum

    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
