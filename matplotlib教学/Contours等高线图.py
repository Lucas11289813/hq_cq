import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 生成二维坐标系
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
# 在坐标系中，选择部分坐标点填充值
Z[1:-1, 1] = 1
Z[1:-1, -2] = 1
Z[1, 1:-2] = 1
Z[-2, 1:-2] = 1
Z[3, 3: 6] = 2
Z[6, 3: 6] = 2
Z[3: 7, 3] = 2
Z[3: 7, 6] = 2
print(Z)

# 绘制等高线
fig, ax = plt.subplots()
# levels: 显示的线的数量
CS = plt.contour(X, Y, Z, levels=3)
ax.clabel(CS, inline=True, fontsize=10)
plt.colorbar(label='Z')
plt.show()

# # 生成数据
# x = np.arange(100)
# y = np.arange(100)
# X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)
# Z[20:-20, 20:-20] = 1
# Z[40:-40, 40:-40] = 2
# print(Z)
#
# # 绘制等高线
# fig, ax = plt.subplots()
# CS = plt.contour(X, Y, Z, levels=1)
# ax.clabel(CS, inline=True, fontsize=10)
# plt.colorbar(label='Z')
# plt.show()
