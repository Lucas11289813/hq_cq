import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['YouYuan']

# # 二维平面改造成三维
#
# x = np.arange(10)
# y = np.random.randint(-10, 10, len(x))
# print(y.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # 分别设置 x y z 轴上的值
# ax.plot(x, np.zeros(len(x)), y)
#
# plt.show()


# 三维点图
_x = np.arange(20)
_y = np.arange(20)
x, y = np.meshgrid(_x, _y)
z = (x - y) ** 2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z)

plt.show()

# # 3D bar
#
# # set up the figure and Axes
# fig = plt.figure(figsize=(8, 3))
# # 121: 第一个数表示行，第二个数表示列，第三个数表示第几个，此处意思是有1行2列的图表，ax1是第一个图
# # 122: 1行2列第二个图表
# # projection='3d': 投影模式为3d
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
#
# # 假数据
# # 横坐标
# _x = np.arange(4)
# # 纵坐标
# _y = np.arange(5)
# # 创建二维坐标系
# _xx, _yy = np.meshgrid(_x, _y)
# # 将坐标系展平，作为bar3d输入
# x, y = _xx.ravel(), _yy.ravel()
#
# # 每个柱状体的顶部值
# top = x + y
# # 每个柱状体的底部值
# bottom = np.zeros_like(top)
# # 柱状体宽度和深度
# width = depth = 1
#
# # shade: 是否开启阴影
# ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
# ax1.set_title('Shaded')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
#
# ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
# ax2.set_title('Not Shaded')
#
# plt.show()
