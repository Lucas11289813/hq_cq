import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['YouYuan']

x = np.arange(50)
y = np.array([np.random.randint(-10, 10) for i in x])

fig, ax = plt.subplots()
line, = ax.plot(x, y)

# 设置坐标轴的范围
ax.set_xlim(-100, 100)
ax.set_ylim(-20, 20)

# 设置坐标轴名称
ax.set_xlabel('横轴')
ax.set_ylabel('纵轴')

# 设置标题
ax.set_title('这是标题')

plt.show()
