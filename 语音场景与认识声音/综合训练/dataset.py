import torchaudio

# 读取音频
waveform, sample_rate = torchaudio.load('./独坐敬亭山.wav')

# 词库
vocab = '- 众 鸟 高 飞 尽 孤 云 独 去 闲 相 看 两 不 厌 只 有 敬 亭 山'.split(' ')
# 文本的索引字典
dictionary = {word: i for i, word in enumerate(vocab)}

# 对应音频的文本
label = '众 鸟 高 飞 尽 孤 云 独 去 闲 相 看 两 不 厌 只 有 敬 亭 山'.split(' ')

# 把标签转换成索引用于后续训练
tokens = [dictionary[word] for word in label]
