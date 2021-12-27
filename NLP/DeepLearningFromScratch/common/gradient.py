import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp_x = x[i]
        x[i] = tmp_x + h
        fhx1 = f(x)

        x[i] = tmp_x - h
        fhx2 = f(x)

        grad[i] = (fhx1 - fhx2) / (2 * h)
        x[i] = tmp_x

    return grad