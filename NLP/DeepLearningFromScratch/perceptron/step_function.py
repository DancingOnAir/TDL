import numpy as np
import matplotlib.pyplot as plt


def step_function1(x):
    y = x > 0
    return y.astype(int)


def step_function2(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_step_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    pass


if __name__ == '__main__':
    test_step_function()



