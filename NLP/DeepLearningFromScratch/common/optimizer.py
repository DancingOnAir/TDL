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
            for k, val in params.item():
                self.v[k] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = dict()
            for k, val in params.item():
                self.h[k] = np.zeros_like(val)

            for k in params.keys():
                self.h[k] += grads[k] * grads[k]
                params -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = dict()
            for k, val in params.item():
                self.h[k] = np.zeros_like(val)

        for k in params.keys():
            self.h[k] *= self.decay_rate
            self.h[k] += (1 - self.decay_rate) * grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + 1e-7)

# https://www.zhihu.com/question/323747423/answer/790457991
# https://blog.csdn.net/zk_ken/article/details/82416061 这里有公式推导
class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.iter = 0

    def update(self, params, grads):
        if self.momentum is None:
            for k, val in params.item():
                self.v = dict()
                self.v[k] = np.zeros_like(val)

                self.m = dict()
                self.m[k] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for k in params.keys():
            self.m[k] += (1 - self.beta1) * (grads[k] - self.m[k])
            self.v[k] += (1 - self.beta2) * (grads[k] ** 2 - self.v[k])

            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + 1e-7)