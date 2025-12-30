import torch
import matplotlib.pyplot as plt

x1 = torch.linspace(-10, 10, 40)
x2 = torch.linspace(-10, 10, 40)

w1 = torch.tensor(0.1)
w2 = torch.tensor(0.2)
w3 = torch.tensor(-1.)
w4 = torch.tensor(-1.)
b1 = torch.tensor(1.)
b2 = torch.tensor(1.)

x1, x2 = torch.meshgrid((x1, x2), indexing='ij')


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# 隐藏层只有一个神经元
y1 = sigmoid(x1 * w1 + x2 * w2 + b1)
# 隐藏层有两个神经元
y2 = sigmoid(x1 * w1 + x2 * w2 + b1) + sigmoid(x1 * w3 + x2 * w4 + b2)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(x1, x2, y1, cmap=plt.cm.YlGnBu_r)
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(x1, x2, y2, cmap=plt.cm.YlGnBu_r)
plt.show()
