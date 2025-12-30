import torch
import matplotlib.pyplot as plt

fig = plt.figure()
w1 = torch.arange(-1, 1, 0.01)
w2 = torch.arange(-1, 1, 0.01)

ax1 = fig.add_subplot(111)
x, y = torch.meshgrid((w1, w2), indexing='ij')
loss = x ** 2 + 2 * y ** 2
ax1.contour(x, y, loss)

# 自适应梯度下降法 --- AdaGrad
epochs = 20
points = []
w1_pre = 1
w2_pre = 0.8
# 注意：S起始值需要稍微大一点（原因，除数的因子太小，学习率会无限大）
S_w1 = 2
S_w2 = 2
epsilon = 1e-3
w1_quard = 0
w2_quard = 0
for epoch in range(epochs):
    e = w2_pre ** 2 + 2 * w2_pre ** 2
    points.append([w1_pre, w2_pre])
    w1_pre = w1_pre - 0.1 / (S_w1 + epsilon) * 2 * w1_pre
    w2_pre = w2_pre - 0.1 / (S_w2 + epsilon) * 4 * w2_pre
    w1_quard += (2 * w1_pre) ** 2
    w2_quard += (4 * w2_pre) ** 2
    S_w1 = w1_quard ** 0.5
    S_w2 = w2_quard ** 0.5

points = torch.tensor(points)
ax1.plot(points[:, 0], points[:, 1], 'ro-')

plt.show()
