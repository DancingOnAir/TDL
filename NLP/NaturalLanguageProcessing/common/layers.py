import numpy as np


class MatMul:
    def __init__(self, W, b=0):
        self.W = W
        self.b = b

        self.x = None
        self.dx = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        self.dW = dout * self.x
        self.dx = dout * self.W

        return self.dW, self.dx
