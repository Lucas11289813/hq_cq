import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from model import w2v2
from dataset import waveform, dictionary, sample_rate, vocab

# 音频总时长
times = waveform.shape[1] / sample_rate

# 加载模型
w2v2.load_state_dict(torch.load('model.pt'))
w2v2.eval()

# 音频提取特征
# 梅尔频率倒谱系数 转换器
transform = T.MFCC(
    sample_rate=sample_rate,
    # 变换器输出的个数，类似卷积的输出通道数
    # n_mfcc 不能大于 n_mels
    n_mfcc=23,
    # n_mels: 梅尔滤波器的个数
    melkwargs={'n_fft': 400, 'win_length': 256, 'hop_length': 256, 'n_mels': 23, 'power': 2}
)

# 提取特征
data = transform(waveform)
data = data.reshape(1, -1)

# 预测结果
with torch.inference_mode():
    emission, _ = w2v2(data, None)

print(emission.shape)

# 计算时间间隔
timespan = times / emission.shape[1]

# 求概率分布
r = emission.softmax(-1)
# 求最大值索引
idx = r.argmax(-1)
print(idx)

# 过滤预测结果，形成真实索引
tokens = []
idx = idx[0].numpy()
last_idx = None
for i in idx:
    if i == last_idx:
        continue
    last_idx = i
    if i == 0:
        continue
    tokens.append(i)

# 用于对齐时的标签
print(tokens)


def align(_emission, _token):
    _token = torch.tensor([_token])
    alignments, scores = F.forced_align(_emission, _token, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.sigmoid()
    return alignments, scores


# 对齐音频和文本
alignments, scores = align(emission, tokens)
print(scores)
print(alignments.shape)

# 融合
token_spans = F.merge_tokens(alignments, scores)

print(token_spans)

print(f'{"":<8}{"开始":<8}{"结束":<8}')
for token_span in token_spans:
    print(f'{vocab[token_span.token]:<8}{token_span.start * timespan:<8.2f}{token_span.end * timespan:<8.2f}')
