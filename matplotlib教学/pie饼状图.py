import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['YouYuan']


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = [1, 2, 3]
explode = [.1, .2, .3]

fig, ax = plt.subplots()


def pct(num):
    return round(num, 2)


# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html
# 基本用法
# x: 值
# explode: 间距
# hatch: 饼状图的花纹 带选项 hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
# autopct: 显示的百分比数字
# pctdistance: 文字的相对距离 0~1 在圆内，大于 1 圆外
# labels: 标签名，影响图例中的显示
# labeldistance: 标签距离
# shadow: 阴影
# startangle: 起始角度
# radius: 半径
# counterclock: 顺时针还是逆时针显示

# patches 是 Wedge 类实例: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Wedge.html#matplotlib.patches.Wedge
patches, texts, autotexts = ax.pie(x, explode, colors=['r', 'y', 'b'], hatch=['/', '\\', '*'],
                                   autopct=pct, pctdistance=1.2, labels=['label1', 'label2', 'label3'],
                                   labeldistance=0.6,
                                   # shadow=True)
                                   # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Shadow.html#matplotlib.patches.Shadow
                                   shadow={
                                       # 颜色饱和度
                                       'shade': 1,
                                       # 阴影偏移
                                       'ox': -0.1,
                                       'oy': 0.2
                                   },
                                   startangle=30, radius=1.2, counterclock=False,
                                   # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Wedge.html#matplotlib.patches.Wedge
                                   # 饼图样式
                                   wedgeprops={
                                       'edgecolor': '#ff00ff',
                                       'linewidth': 5,
                                       'linestyle': '--'
                                   },
                                   textprops={
                                       'color': '#ffffff',
                                       'size': 24
                                   },
                                   # center: 中心坐标
                                   # frame: 是否显示轴框
                                   # rotatelabels: 旋转label
                                   center=(10, 10), frame=True, rotatelabels=True)

print(patches)
print(texts)
print(autotexts)

plt.legend()
plt.show()
