import sys
import os
sys.path.append(os.pardir)

import time
import numpy
import matplotlib as plt
from common.np import *
from common.util import *


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        pass

    def fit(self, x, t, epoch_num=10, batch_size=32, grad=None, eval_interval=20):
        data_size = len(x)
        iter_num = data_size // batch_size
        model = self.model
        optimizer = self.optimizer
        total_loss = 0

        start_time = time.time()

        pass

    def plot(self):
        pass