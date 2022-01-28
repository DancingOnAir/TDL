import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = dict()
            for k, val in params.items():
                self.v[k] = np.zeros_like(val)

        for k in params.keys():
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = dict()
            for k, val in params.items():
                self.h[k] = np.zeros_like(val)

        for k in params.keys():
            self.h[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, parmas, grads):
        if self.h is None:
            self.h = dict()
            for k, val in parmas.items():
                self.h[k] = np.zeros_like(val)

        for k in parmas.keys():
            self.h[k] *= self.decay_rate
            self.h[k] += (1 - self.decay_rate) * grads[k] * grads[k]
            parmas[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = None
        self.v = None
        self.iter = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = dict()
            self.v = dict()

            for k, val in params.items():
                self.m[k] = np.zeros_like(val)
                self.v[k] = np.zeros_like(val)

        self.iter += 1
        lr_iter = self.lr * np.sqrt(1 - self.beta2 ** self.iter) / (1 - self.beta1 ** self.iter)

        for k in params.keys():
            self.m[k] += (1 - self.beta1) * (grads[k] - self.m[k])
            self.v[k] += (1 - self.beta2) * (grads[k] ** 2 - self.v[k])
            params[k] -= lr_iter * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)
