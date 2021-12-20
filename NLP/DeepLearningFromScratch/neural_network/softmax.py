import numpy as np


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    exp_a_sum = np.sum(exp_a)
    y = exp_a / exp_a_sum

    return y


def test_softmax():
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    print(sum(y))


if __name__ == '__main__':
    test_softmax()
