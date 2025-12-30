import librosa
import torchaudio
from matplotlib import pyplot as plt
import torchaudio.transforms as T


# 绘制频率分贝图
# 横轴是 T.Spectrogram 窗口滑动的次数，纵轴是频率，被分成了 bin = n_fft // 2 + 1 个块
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # librosa.power_to_db(specgram): 将幅度谱转换为分贝谱
    # 纵坐标是频率，横坐标是时间（也就是帧）
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


# 加载音频
waveform, sample_rate = torchaudio.load('./audios/1688-142285-0007.wav')
print(waveform.shape)
print(sample_rate)

# 短时傅里叶变换转换器
# n_fft: 影响频率轴切分成多少个 bin，bin = n_fft // 2 + 1，bin是一个频率块，n_fft越大，频率图中频率分辨率越高
# win_length: 窗口长度，默认是 n_fft，是时间轴上的窗口大小
# hop_length: 窗口滑动步长，默认是 n_fft // 4，是时间轴上的窗口滑动步长
transform = T.Spectrogram(n_fft=64, win_length=64, hop_length=64, power=2)
result = transform(waveform)
print(result.shape)

# 显示图像
# plot_spectrogram(result[0])

# 梅尔频率倒数变换
# T.MFCC 比 T.MelSpectrogram 多一个对数过程，结果为对数值
# 科学家总结: 人耳听到的频率特征符合对数的特征
transform = T.MFCC(
    sample_rate=sample_rate,  # 采样率
    n_mfcc=13,  # 梅尔滤波器的个数 类似于卷积的输出通道
    melkwargs={'n_fft': 2048, 'win_length': 128, 'hop_length': 128, 'power': 2}
)

result = transform(waveform)
print(result.shape)

plot_spectrogram(result[0])

plt.show()
