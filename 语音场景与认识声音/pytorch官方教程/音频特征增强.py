import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import pyaudio

SAMPLE_WAV_SPEECH_PATH = '../Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'


# def _get_sample(path, resample=None):
#     effects = [["remix", "1"]]
#     if resample:
#         effects.extend(
#             [
#                 ["lowpass", f"{resample // 2}"],
#                 ["rate", f"{resample}"],
#             ]
#         )
#     return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def _get_sample(path, resample=None):
    waveform, sample_rate = torchaudio.load(path)
    if resample:
        # 如果需要重新采样，使用 torchaudio.transforms.Resample 进行调整
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample)
        waveform = resampler(waveform)
        sample_rate = resample
    return waveform, sample_rate


def get_speech_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)


def get_spectrogram(
        n_fft=400,
        win_len=None,
        hop_len=None,
        power=2.0,
):
    waveform, _ = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


def preview(spec, rate=16000):
    ispec = T.InverseSpectrogram()
    waveform = ispec(spec)

    # 创建 PyAudio 对象
    p = pyaudio.PyAudio()

    # 打开流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=int(rate),
                    output=True)

    # 将 tensor 转换为 numpy 数组，并将数据类型转换为浮点数
    waveform_np = waveform.numpy().astype('float32')

    # 写入流并播放
    stream.write(waveform_np.tobytes())

    # 停止流
    stream.stop_stream()
    stream.close()

    # 关闭 PyAudio
    p.terminate()


#######################################################
# 时间缩放
#######################################################

# spec = get_spectrogram(power=None)
# stretch = T.TimeStretch()
#
# spec_12 = stretch(spec, overriding_rate=3)
# spec_09 = stretch(spec, overriding_rate=0.3)
#
#
# def plot():
#     def plot_spec(ax, spec, title):
#         ax.set_title(title)
#         ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")
#
#     fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
#     plot_spec(axes[0], torch.abs(spec_12[0]), title="Stretched x1.2")
#     plot_spec(axes[1], torch.abs(spec[0]), title="Original")
#     plot_spec(axes[2], torch.abs(spec_09[0]), title="Stretched x0.9")
#     fig.tight_layout()
#
#
# plot()
# plt.show()

# preview(spec)
# preview(spec_12)
# preview(spec_09)

#######################################################
# 时间频率掩码
#######################################################
torch.random.manual_seed(4)

time_masking = T.TimeMasking(time_mask_param=80)
freq_masking = T.FrequencyMasking(freq_mask_param=80)

spec = get_spectrogram()
time_masked = time_masking(spec)
freq_masked = freq_masking(spec)

# 此处声音无法播放，因为调用 get_spectrogram 时，已经使用了平方操作将复数张量变成了实数张量
# 这个过程是不可逆的
# 而 preview 要播放声音，需要用到复数张量才能逆变换会时域数据
# preview(spec)

def plot():
    def plot_spec(ax, spec, title):
        ax.set_title(title)
        ax.imshow(librosa.power_to_db(spec), origin="lower", aspect="auto")

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    plot_spec(axes[0], spec[0], title="Original")
    plot_spec(axes[1], time_masked[0], title="Masked along time axis")
    plot_spec(axes[2], freq_masked[0], title="Masked along frequency axis")
    fig.tight_layout()


plot()
plt.show()
