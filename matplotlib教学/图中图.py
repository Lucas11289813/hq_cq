import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 指定后端
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['YouYuan']

x = np.arange(10)


def random_y(low, high, count):
    return np.random.randint(low, high, count)


# fig = plt.figure()
# ax = fig.add_subplot()

fig, ax = plt.subplots()

ax.plot(x, random_y(-10, 10, len(x)))

# 绘制图中图
# 添加 Axes 对象，设置坐标点和宽高，所有值都是百分比
subax = fig.add_axes([0.2, 0.65, 0.2, 0.2])
# 绘制数据
subax.plot([1, 2, 3, 4], [10, 20, 25, 30])  # 绘制图中图的内容

plt.show()
