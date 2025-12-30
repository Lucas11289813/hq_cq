import numpy as np
import matplotlib.pyplot as plt


def random_y(low, high, count):
    return np.array([np.random.randint(low, high) for i in range(count)])


x = np.arange(10)

fig, ax = plt.subplots()
# 设置预设的颜色英文
ax.plot(x, random_y(0, 10, len(x)), color='red')
# 设置十六进制颜色值
ax.plot(x, random_y(0, 10, len(x)), color='#00ff00')

plt.show()
