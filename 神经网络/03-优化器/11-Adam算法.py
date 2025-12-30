import torch
import matplotlib.pyplot as plt

fig = plt.figure()
w1 = torch.arange(-1, 1, 0.01)
w2 = torch.arange(-1, 1, 0.01)

ax1 = fig.add_subplot(111)
x, y = torch.meshgrid((w1, w2), indexing='ij')
loss = x ** 2 + 2 * y ** 2
ax1.contour(x, y, loss)

# 二阶动量法 --- Adam ----
epochs = 20
points = []
w1_pre = 1
w2_pre = 0.8
# 梯度动量
V_w1 = 0
V_w2 = 0
# 学习率动量
S_w1 = 2
S_w2 = 2
epsilon = 1e-3
w1_quard = 0
w2_quard = 0
beta = 0.1
for epoch in range(epochs):
    e = w2_pre ** 2 + 2 * w2_pre ** 2
    points.append([w1_pre, w2_pre])
    # 梯度动量累加
    V_w1 = beta * V_w1 + (1 - beta) * (2 * w1_pre)
    V_w2 = beta * V_w2 + (1 - beta) * (4 * w2_pre)
    # 学习率动量累加
    w1_quard = w1_quard * beta + (1 - beta) * (2 * w1_pre) ** 2
    w2_quard = w1_quard * beta + (1 - beta) * (4 * w2_pre) ** 2
    S_w1 = w1_quard ** 0.5
    S_w2 = w2_quard ** 0.5
    # 二阶动量的组合
    w1_pre = w1_pre - 0.1 / (S_w1 + epsilon) * V_w1
    w2_pre = w2_pre - 0.1 / (S_w2 + epsilon) * V_w2

points = torch.tensor(points)
ax1.plot(points[:, 0], points[:, 1], 'ro-')

plt.show()
