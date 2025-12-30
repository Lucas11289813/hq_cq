import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from torchaudio.utils import download_asset

torch.random.manual_seed(0)

SAMPLE_SPEECH = '../Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # 将频谱图的纵坐标单位转换成分贝
    # 默认 T.Spectrogram 的 power=None 则转换结果单位是振幅(amplitude)，power=2 时，单位是功率(power)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


# 加载声音
SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)
print(SPEECH_WAVEFORM.shape)
print(SAMPLE_RATE)

############################################################
# 显示频谱图
############################################################


# # 频谱变换
# spectrogram = T.Spectrogram(n_fft=512)
#
# # 执行转换
# spec = spectrogram(SPEECH_WAVEFORM)
#
# fig, axs = plt.subplots(2, 1)
# plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
# plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
# fig.tight_layout()
# plt.show()

############################################################
# 不同傅里叶窗口分辨率的效果
############################################################

# n_ffts = [32, 128, 512, 2048]
# hop_length = 64
#
# specs = []
# for n_fft in n_ffts:
#     spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
#     spec = spectrogram(SPEECH_WAVEFORM)
#     print(spec.shape)
#     specs.append(spec)
#
# fig, axs = plt.subplots(len(specs), 1, sharex=True)
# for i, (spec, n_fft) in enumerate(zip(specs, n_ffts)):
#     plot_spectrogram(spec[0], ylabel=f"n_fft={n_fft}", ax=axs[i])
#     axs[i].set_xlabel(None)
# fig.tight_layout()
# plt.show()

# 结论
# n_fft 的值决定了频率轴的分辨率 (图中纵坐标)。然而，随着 n_fft 值越高，能量将分布在更多的 bins 中 (这里指的是纵坐标)，
# 因此当您可视化它时，即使它们的分辨率更高，它也可能看起来更模糊。

############################################################
# 不同采样率的效果
############################################################

# # 下采样到原来的一半
# speech2 = torchaudio.functional.resample(SPEECH_WAVEFORM, SAMPLE_RATE, SAMPLE_RATE // 2)
# # 上采样回原来的采样率
# speech3 = torchaudio.functional.resample(speech2, SAMPLE_RATE // 2, SAMPLE_RATE)
# # 创建频谱变换
# spectrogram = T.Spectrogram(n_fft=512)
#
# spec0 = spectrogram(SPEECH_WAVEFORM)
# spec2 = spectrogram(speech2)
# spec3 = spectrogram(speech3)
#
# # 可视化
# fig, axs = plt.subplots(3, 1)
# plot_spectrogram(spec0[0], ylabel="Original", ax=axs[0])
# axs[0].add_patch(Rectangle((0, 3), 212, 128, edgecolor="r", facecolor="none"))
# plot_spectrogram(spec2[0], ylabel="Downsampled", ax=axs[1])
# plot_spectrogram(spec3[0], ylabel="Upsampled", ax=axs[2])
# fig.tight_layout()
# plt.show()

# 结论
# 采样率会决定纵轴 频率轴的每个单位长度的含义不同，可以认为是频率轴的分辨率不同

############################################################
# 林格里芬 恢复波形图
############################################################

# 要从频谱图中恢复波形，您可以使用 torchaudio.transforms.GriffinLim 。

# # 定义变换
# n_fft = 1024
# spectrogram = T.Spectrogram(n_fft=n_fft)
# griffin_lim = T.GriffinLim(n_fft=n_fft)
#
# # 应用变换
# spec = spectrogram(SPEECH_WAVEFORM)
# reconstructed_waveform = griffin_lim(spec)
#
# _, axes = plt.subplots(2, 1, sharex=True, sharey=True)
# plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original", ax=axes[0])
# plot_waveform(reconstructed_waveform, SAMPLE_RATE, title="Reconstructed", ax=axes[1])
# plt.show()

############################################################
# 梅尔滤波器组
############################################################

# torchaudio.functional.melscale_fbanks() 生成用于将频率仓转换为梅尔尺度仓的滤波器组。

# n_fft = 512
# n_mels = 64
# sample_rate = 16000
#
# # 创建滤波器
# mel_filters = F.melscale_fbanks(
#     int(n_fft // 2 + 1),
#     n_mels=n_mels,
#     f_min=0.0,
#     f_max=sample_rate / 2.0,
#     sample_rate=sample_rate,
#     norm="slaney",
# )
# plot_fbank(mel_filters, "Mel Filter Bank - torchaudio")
#
# plt.show()

# 结论
# 这里显示了梅尔滤波器组，这个滤波器是非线性的，用于模拟人耳听声音的过程。
# 梅尔频谱特征提取的过程中，会使用这个滤波器和傅里叶变换的频域图进行点积
# 梅尔滤波器可以理解成掩码，和短时傅里叶变换的频谱图进行点积，提取频谱图中，人类能够识别的部分


############################################################
# 梅尔频谱图
############################################################

# n_fft = 1024
# win_length = None
# hop_length = 512
# n_mels = 128
#
# # 声明一个梅尔频谱转换器
# mel_spectrogram = T.MelSpectrogram(
#     sample_rate=SAMPLE_RATE,
#     n_fft=n_fft,
#     win_length=win_length,
#     hop_length=hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     norm="slaney",
#     n_mels=n_mels,
#     mel_scale="htk",
# )
#
# melspec = mel_spectrogram(SPEECH_WAVEFORM)
# plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
# plt.show()

############################################################
# 梅尔频率倒谱系数
############################################################

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)
print(mfcc)
print(mfcc.shape)
plot_spectrogram(mfcc[0], title="MFCC")
plt.show()

############################################################
# 线性频率倒谱系数
############################################################

# n_fft = 2048
# win_length = None
# hop_length = 512
# n_lfcc = 256
#
# lfcc_transform = T.LFCC(
#     sample_rate=SAMPLE_RATE,
#     n_lfcc=n_lfcc,
#     speckwargs={
#         "n_fft": n_fft,
#         "win_length": win_length,
#         "hop_length": hop_length,
#     },
# )
#
# lfcc = lfcc_transform(SPEECH_WAVEFORM)
# plot_spectrogram(lfcc[0], title="LFCC")
# plt.show()
