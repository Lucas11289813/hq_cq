import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

matplotlib.use('TkAgg')
# plt.rcParams['font.sans-serif'] = ['YouYuan']

fig, ax = plt.subplots()

count = 0
x = []
y = []

line, = ax.plot(x, y)


def animate(frame):
    global fig, x, y, count

    # frame: 下一帧的值
    # print(frame)
    x.append(count)
    count += np.deg2rad(1)
    y.append(frame)
    line.set_data(x, y)
    if len(x) > 2:
        ax.set_xlim(min(x), max(x))
    else:
        ax.set_xlim(0, 1)
    if len(y) > 2:
        ax.set_ylim(min(y), max(y))
    else:
        ax.set_ylim(0, 1)

    fig.canvas.draw()


# 创建动画
ani = animation.FuncAnimation(
    fig,
    # 更新画面的动画函数
    animate,
    # 动画中每帧的数据集，可以是可迭代对象，也可以是函数，用于提供迭代数据
    np.sin(np.linspace(0, 2 * np.pi * 3, 91 * 3)),
    # 动画时间间隔
    interval=16,
    # 是否重复
    repeat=True
)

plt.show()
