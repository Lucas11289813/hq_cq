import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 获取系统上安装的所有字体
font_list = sorted([f.name for f in font_manager.fontManager.ttflist])

# 打印可用的字体列表
for font in font_list:
    print(font)

# 设置字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.sans-serif'] = ['LiSu']
# plt.rcParams['font.sans-serif'] = ['LiSu']
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['YouYuan']

x = np.linspace(0, 2, 100)

# 显示绘图:
# 显示创建图像对象，并显示
# 创建子图集，这里只有一个图
fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
# 给轴添加参数
ax.plot(x, x, label='直线')  # 设置一些数据到轴上
ax.plot(x, x ** 2, label='平方')  # 再设置一组数据到轴上
ax.plot(x, x ** 3, label='立方')  # 更多的数据
ax.set_xlabel('x 轴')  # 添加x轴名称
ax.set_ylabel('y 轴')  # 添加y轴名称
ax.set_title('这是标题')  # 标题
ax.legend()  # 添加图例
ax.grid()  # 添加网格

plt.show()

# 隐式绘图:
# 代码底层会隐式调用对应的创建图像的接口
plt.figure(figsize=(4, 4))
plt.plot(x, x, label='linear')
plt.plot(x, x ** 2, label='quadratic')
plt.plot(x, x ** 3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()

plt.show()
