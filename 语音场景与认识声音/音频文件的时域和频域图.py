import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt

# 加载音频
# 返回值:
# waveform: 所有声道的时域数据
# sample_rate: 采样率，代表 1秒中采样多少个样本
waveform, sample_rate = torchaudio.load('./audios/1688-142285-0007.wav')
print(waveform)
# waveform 的形状 (channels, times)
# channels: 声道
# times: 每个时刻的值
print(waveform.shape)
print(sample_rate)

# 音频的时长
time = waveform.shape[1] / sample_rate
print(time)

# 时域图
# 时间轴，单位 秒
t = np.linspace(0, time, waveform.shape[1])

# 频域图
# 振幅
a = torch.abs(torch.fft.fft(waveform[0]))
a = a[:len(a) // 2]

# 频率
f = torch.fft.fftfreq(waveform.shape[1], 1 / sample_rate)
f = f[:len(f) // 2]

fig, ax = plt.subplots(2)

ax[0].plot(t, waveform[0].numpy())
ax[1].plot(f.numpy(), a.numpy())

plt.show()
