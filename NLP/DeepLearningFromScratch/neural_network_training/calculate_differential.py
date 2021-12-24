import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_diff1(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h


def func1(x):
    return 0.01 * x ** 2 + 0.1 * x


def test_calculate_diff():
    print(np.float32(1e-50))

    x = np.arange(0.0, 20.0, 0.5)
    y = func1(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print(numerical_diff(func1, 5))
    print(numerical_diff(func1, 10))


if __name__ == '__main__':
    test_calculate_diff()
