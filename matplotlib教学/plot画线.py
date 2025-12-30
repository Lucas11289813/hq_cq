import numpy as np
import matplotlib.pyplot as plt


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(10)

fig, ax = plt.subplots()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
# 基础画线
# 返回列表的第一个值为Line2D对象列表
# Line2D 对应的就是线对象，参考: https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
# line1, = ax.plot(x, random_y(0, 10, len(x)))
# line2, = ax.plot(x, random_y(0, 10, len(x)))
# plt.show()

# 格式
# 第三个参数代表格式化字符串，形式为：fmt = '[marker][line][color]'
# marker: 代表点形状
# line: 线样式
# color: 线颜色
# 具体标记怎么写，可以参考官网: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot
# ax.plot(x, random_y(0, 10, len(x)), ',:y')
# ax.plot(x, random_y(0, 10, len(x)), 'd-.m')
# plt.show()

# 同时添加多条线
# lines = ax.plot(x, random_y(0, 5, len(x)), 'o-r', x, random_y(0, 10, len(x)), 'v--b')
# print(lines)
# plt.show()


# 还可以再末尾添加 Line2D 的关键字参数
# 例如: 线条标签（用于自动图例）、线宽、抗锯齿、标记面颜色等属性
# 标记预览: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
ax.plot(random_y(0, 10, 6), label='line1', color=(1, 0, 1), linewidth=1, marker='d', linestyle='--', alpha=0.3)
ax.plot(random_y(0, 10, 6), label='line2', color='#ffff00', linewidth=2, marker='o', linestyle='-.',
        markerfacecolor='#0000ff')
plt.legend()

plt.show()
