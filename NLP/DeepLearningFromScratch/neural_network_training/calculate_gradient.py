import numpy as np
import matplotlib.pyplot as plt


def func2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # np.array.size: number of elements in the array, equal to np.prod(a.shape).
    for i in range(x.size):
        tmp_x = x[i]
        x[i] = tmp_x + h
        fxh1 = f(x)

        x[i] = tmp_x - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_x

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_list = [x[0]]
    y_list = [x[1]]

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

        x_list.append(x[0])
        y_list.append(x[1])

    plt.scatter(x_list, y_list, step_num)
    plt.xlim(-3.1, 3.1)
    plt.ylim(-4.1, 4.1)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()
    return x


def test_numerical_gradient():
    print(numerical_gradient(func2, np.array([3.0, 4.0])))
    print(numerical_gradient(func2, np.array([0.0, 2.0])))
    print(numerical_gradient(func2, np.array([3.0, 0.0])))

    init_x = np.array([-3.0, 4.0])
    y = gradient_descent(func2, init_x, lr=0.1, step_num=100)
    print(y)


if __name__ == '__main__':
    test_numerical_gradient()
