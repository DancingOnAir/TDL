import numpy as np


def mean_squared_error(y, t):
    print(y - t)
    return 0.5 * np.sum((y - t) ** 2)


# for y.ndim = 1
def cross_entropy_error(y, t):
    print(y * t)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# for one-hot
def cross_entropy_error_batch_one_hot(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


# for non one-hot, actual value
def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def test_lost_function():
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    mse = mean_squared_error(np.array(y), np.array(t))
    print("mse: " + str(mse))

    cee = cross_entropy_error(np.array(y), np.array(t))
    print("cee: " + str(cee))


if __name__ == '__main__':
    test_lost_function()

