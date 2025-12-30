import torch
import matplotlib.pyplot as plt

fig = plt.figure()
w1 = torch.arange(-1, 1, 0.01)
w2 = torch.arange(-1, 1, 0.01)

ax1 = fig.add_subplot(121, projection="3d")
x, y = torch.meshgrid((w1, w2), indexing='ij')
loss = x ** 2 + 2 * y ** 2
ax1.plot_surface(x, y, loss, cmap=plt.cm.YlGnBu_r)

ax2 = fig.add_subplot(122)
ax2.contour(x, y, loss)
plt.show()
