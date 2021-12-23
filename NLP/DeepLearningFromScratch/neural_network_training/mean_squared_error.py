import numpy as np


def mean_squared_error(y, t):
    print(y - t)
    return 0.5 * np.sum((y - t) ** 2)


def test_lost_function():
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    mse = mean_squared_error(np.array(y), np.array(t))
    print("mse: " + str(mse))


if __name__ == '__main__':
    test_lost_function()

