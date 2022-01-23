import sys
import os
sys.path.append(os.pardir)


import numpy as np
from common.layers import MatMul


def test_np_dot():
    c = np.array([[1, 0, 0, 0, 0, 0, 0]])
    W = np.random.randn(7, 3)
    print(W)

    print('np.dot')
    h = np.dot(c, W)
    print(h)
    print('-' * 50)

    print('MatMul layer')
    layer = MatMul(W)
    h = layer.forward(c)
    print(h)


if __name__ == '__main__':
    test_np_dot()
