import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['YouYuan']


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(4)

fig, ax = plt.subplots()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html
# 基础用法
y = random_y(5, 15, len(x))

# ax.plot(x, y, '^--r')

# 返回值 BarContainer: https://matplotlib.org/stable/api/container_api.html#matplotlib.container.BarContainer
# width: 宽度
# bottom: 底部值
# align: 对齐方式
# bar_container = ax.bar(x, y, width=0.2, bottom=-2, align='center')
# bar_container = ax.bar(x, y,
#                        width=[0.2, 0.4, 1, 0.5, 0.2, 1.2],
#                        bottom=[-1, 1, -3, -5, -2, 0],
#                        align='edge')
#
# plt.show()

# 颜色 线宽
# color: 主柱颜色
# edgecolor: 边框颜色
# linewidth: 边框宽度
# tick_label: 刻度标签
# label: 图例用的标签，也是分组标签
# xerr, yerr: 误差
# ecolor: 误差线颜色
# ax.bar(x, y, color='y', edgecolor='g', linewidth=5, tick_label='bar label', label='label',
#        xerr=0.2, yerr=0.4, ecolor='r')

# ax.bar(x, y, width=0.4, label='bar demo',
#        color=[(1, 0, 0), '#ffff00', 'b', '#00ff00'],
#        edgecolor=['g', 'b', 'y', 'r'],
#        linewidth=[1, 2, 3, 4],
#        tick_label=['电器', '日化', '服饰', '零食'],
#        # yerr 写法相同
#        xerr=[
#            # 第一行代表 - 误差
#            [.1, .2, .3, .4],
#            # 第二行代表 + 误差
#            [.4, .3, .2, .1]
#        ],
#        ecolor=['r', 'y', 'b', 'g'])
#
# plt.legend()
# plt.show()

# 其他 Rectangle 属性 https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle
# alpha: 透明度
# angle: 逆时针旋转角度
# xy: 左下角锚点坐标
# zorder: 重叠顺序，优先绘制数字小的
# y = [1, 2, 3, 4]
# ax.bar(x, y, width=[.2, .3, .4, .5], color=['r', 'y', 'b', 'g'],
#        alpha=1, angle=-10, xy=(0, 0), zorder=1)
# ax.bar(x, y, zorder=0.5)
#
# plt.show()


# 水平柱状图
# 其他参数和垂直柱状图一样
ax.barh(x, y)
plt.show()
