import numpy as np
import matplotlib.pyplot as plt


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(10)

fig, ax = plt.subplots()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
# 基础散点图
# 返回 PathCollection 对象 https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.PathCollection
# 第三个参数: 点大小
# 第四个参数: 颜色
# scatter1 = ax.scatter(x, random_y(0, 10, len(x)), 30., 'y')
# # 第三个参数可以是列表
# # 第四个参数可以是列表
# scatter2 = ax.scatter(x, random_y(0, 10, len(x)),
#                       [60, 120, 60, 120, 30, 240, 50, 120, 30, 60],
#                       # '#0000ff')
#                       [(1, 0, 0), '#00ff00', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'y'])
# print(scatter1)
# plt.show()

# 颜色映射表
# 创建随机数据
x = np.random.rand(100)
y = np.random.rand(100)
sizes = np.random.rand(100) * 1000  # 数据点大小
colors = np.random.rand(100)  # 数据点颜色

# 热力图
# 前提: c 数量需要和样本点数量相同，且指定 0~1 之间的数字
# cmap: 影响热力图的颜色，带选项: 'viridis'、'coolwarm'、'summer' 等
# 也可以填入 Colormap 对象: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
# 预设的 Colormap 对象: https://matplotlib.org/stable/gallery/color/colormap_reference.html
plt.scatter(x, y, s=sizes, c=colors, cmap='viridis')  # 使用 'viridis' 颜色映射
plt.colorbar()  # 添加颜色条

plt.show()
