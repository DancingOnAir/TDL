import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def test_relu():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)

    plt.plot(x, y)
    plt.ylim(-1.0, 6.0)
    plt.show()


if __name__ == '__main__':
    test_relu()
