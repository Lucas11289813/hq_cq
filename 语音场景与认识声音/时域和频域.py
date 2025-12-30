# 参考视频 https://www.youtube.com/watch?v=fYtVHhk3xJ0&t=3s
# 傅里叶变换参考视频: https://www.bilibili.com/video/BV1pW411J7s8/

import numpy as np
import matplotlib.pyplot as plt

# 时域: 横坐标为时间轴，纵坐标是振幅的音频图像
# 频域: 横坐标为频率轴，纵坐标是振幅的音频图像

t = np.linspace(0, 2, 1000)


# 正弦波公式: a*sin(2*pi*f*t+θ)
# a: 振幅
# f: 频率
# θ: 相位
# t: 自变量横坐标
def sin(a, f, t, θ=0):
    return a * np.sin(2 * np.pi * f * t + θ)


fig, ax = plt.subplots(2, 3, figsize=(9, 6))

# 时域图
line1, = ax[0, 0].plot(t, sin(1, 3, t))
ax[0, 1].plot(t, sin(2, 5, t))
ax[0, 2].plot(t, sin(1, 3, t) + sin(2, 5, t))


# 振幅频率计算函数
# 频域图中，横坐标是波的频率，纵坐标是波的强度，描述波的能量
# wave: 正弦波
def freq(wave):
    # 振幅
    a = np.abs(np.fft.fft(wave))
    # 频率
    # 第一个参数: 样本个数
    # 第二个参数: 采样的时间间隔，采样频率的倒数
    # 此处的 t 代表 2 秒中，有 1000 个数字组成，所以样本为 1000 条，采样率为 1000/2 = 500 hz（代表每秒采样多少个样本）
    freq = np.fft.fftfreq(1000, d=1 / 500)
    return freq[:len(freq) // 2], a[:len(a) // 2]


f, a = freq(sin(1, 3, t))
ax[1, 0].plot(f, a)
f, a = freq(sin(2, 5, t))
ax[1, 1].plot(f, a)
f, a = freq(sin(1, 3, t) + sin(2, 5, t))
ax[1, 2].plot(f, a)

# 设置每幅图的横纵坐标范围
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        if i == 0:
            ax[i, j].set_xlim(0, 2)
            ax[i, j].set_ylim(-4, 4)
        if i == 1:
            ax[i, j].set_xlim(0, 6)

theta = 0
while True:
    theta -= 0.2
    wave = sin(1, 3, t, theta)
    line1.set_ydata(wave)
    fig.canvas.draw()
    plt.pause(0.0167)

plt.show()
