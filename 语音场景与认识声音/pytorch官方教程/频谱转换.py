import torchaudio.transforms as T
import torchaudio

waveform, sample_rate = torchaudio.load('../audios/1688-142285-0007.wav')
print(waveform.shape)

# 频域转换: 提取音频特征
# 转换器
transform = T.Spectrogram(n_fft=64, win_length=64, hop_length=64, power=2)

result = transform(waveform)

print(result.shape)
