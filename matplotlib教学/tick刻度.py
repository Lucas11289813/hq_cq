import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.sans-serif'] = ['YouYuan']


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(4)

fig, ax = plt.subplots()

# 刻度的对数缩放
# data1 = np.random.randn(100)
#
# fig, axs = plt.subplots(1, 2, figsize=(5, 2.7), layout='constrained')
# xdata = np.arange(len(data1))  # make an ordinal for this
# data = 10 ** data1
# axs[0].plot(xdata, data)
#
# axs[1].set_yscale('log')
# axs[1].plot(xdata, data)
#
# plt.show()

ax.plot(x, random_y(0, 10, len(x)))

# 手动设置刻度
# ax.set_xticks(np.arange(0, 4), ['one', 'two', 'three', 'four'])
# ax.set_yticks(np.arange(0, 11), ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十'])


# 自动设置主要次要刻度
# 设置主要刻度的跨度
# 参数为每个刻度的跨度
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter('{x:.0f}')
# 设置次要刻度的跨度
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_formatter('{x:.1f}')

plt.show()
