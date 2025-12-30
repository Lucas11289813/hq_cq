import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['YouYuan']


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(4)
y = [1, 2, 1, 3]

fig, ax = plt.subplots()

ax.plot(x, y)

# api: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate
# 基础用法
# 在注释中，需要考虑两点：被注释数据的位置 xy 和注释文本 xytext 的位置。这两个参数都是 (x, y) 元组
# xy: 要标记的数据坐标
# xytext: 注释位置
# ax.annotate('这是标签', xy=(1, 2), xytext=(1.5, 2.25))


# 切换坐标轴
# 坐标轴带选项，官网可查: https://matplotlib.org/stable/users/explain/text/annotations.html#annotations
# ax.annotate('这是标签', xy=(1, 2), xytext=(0.5, 0.5), xycoords='data', textcoords='axes fraction')


# ha: 水平对齐
# va: 垂直对齐
# ax.annotate('水平对齐\n垂直对齐', xy=(1, 2), xytext=(0.5, 0.5), xycoords='data', textcoords='axes fraction',
#             ha='center', va='top')

# 绘制精美箭头
# 第一二个参数，箭头的起始结束坐标
# mutation_scale: 箭头数据缩放
# arr = mpatches.FancyArrowPatch((1.25, 1.5), (1, 2),
#                                # arrowstyle='->,head_width=.15', mutation_scale=12)
#                                # arrowstyle='-|>,head_width=10,head_length=10', mutation_scale=2)
#                                arrowstyle='-|>', mutation_scale=20)
# # 添加补丁
# ax.add_patch(arr)
# # 将 xycoords 设置为 arr
# ax.annotate('这是标签', xy=(1, 2), xytext=(1.5, 2.25), xycoords=arr)


# 数据文本注释
# ax.scatter(x, y, 50, color='r', zorder=2)
# annotations = ['A', 'B', 'C', 'D']
# for xi, yi, text in zip(x, y, annotations):
#     ax.annotate(text,
#                 xy=(xi, yi), xycoords='data',
#                 xytext=(3, 3), textcoords='offset points', color='#f0f')


# 用内置箭头注释
ax.annotate('这是标签', xy=(1, 2), xytext=(2.5, 3),
            # 箭头属性
            arrowprops=dict(
                # 箭头身体宽度
                # width=5,
                # 箭头头部宽度
                # headwidth=9,
                # 箭头两端距离文本和数据点的间距百分比
                # 该值越大，箭头距离两端的空白越多
                # shrink=0.05,

                # 以下属性请查看: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
                # 也可以查看annotation自定义箭头部分: https://matplotlib.org/stable/users/explain/text/annotations.html#annotations
                # 箭头样式
                arrowstyle='->,head_width=1,head_length=2',
                # arrowstyle='fancy',
                # 连接属性
                # connectionstyle='angle3',
                # connectionstyle='angle',
                connectionstyle='bar,fraction=0.3',

                # 还可以设置 Polygon 中的属性
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html#matplotlib.patches.Polygon
                facecolor='#00ff00',
                edgecolor='#00ff00'
            ))

plt.show()
