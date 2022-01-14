import sys
import os
sys.path.append(os.pardir)

from dataset.spiral import load_data
from matplotlib import pyplot as plt
import numpy as np


def show_spiral_dataset():
    x, t = load_data()
    print(x.shape)
    print(t.shape)

    N = 100
    CLS_NUM = 3
    makers = ['o', 'x', '+']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N: (i+1)*N, 0], x[i*N: (i+1)*N, 1], s=40, marker=makers[i])
    plt.show()


if __name__ == '__main__':
    show_spiral_dataset()
