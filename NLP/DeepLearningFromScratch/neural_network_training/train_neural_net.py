import sys
import os
sys.path.append(os.pardir)


import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
train_loss_list = list()
train_acc_list = list()
test_acc_list = list()

iters_num = 10000
batch_size = 100
train_size = train_x.shape[0]
learning_rate = 0.1
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_x[batch_mask]
    t_batch = train_t[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for k in ('W1', 'b1', 'W2', 'b2'):
        network.params[k] -= learning_rate * grad[k]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_batch, t_batch)
    test_acc = network.accuracy(test_x, test_t)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print('train acc, test acc | ' + str(train_acc) + ", " + str(test_acc))


print('draw plot')
x = list(range(1, iters_num+1))
y = train_loss_list
plt.plot(x, y)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

