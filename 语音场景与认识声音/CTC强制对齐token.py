# CTC 是 "Connectionist Temporal Classification"（连接主义时序分类）的缩写。CTC 是一种用于序列建模的技术，最初被用于语音识别，但后来也被应用于其他序列标记任务，如手写识别和自然语言处理。
# https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

# 总结步骤:
# 1. 准备音频数据
# 2. 准备 wav2vec2 模型
# 3. 使用 wav2vec2 模型提取音频特征 emissions
# 4. 使用 CTC 强制对齐 API 对齐音频 emissions 和文本 token
# 5. 使用令牌融合，将 CTC 强制对齐的内容做融合

# 什么时候用对齐和融合？
# 1. 当需要准确知道每个字的音频出现和结束的时间点，此时需要进行对齐，例如: 自动生成字幕等
# 2. 损失函数 CTCLoss


import torch
import torchaudio
from torchaudio.models import wav2vec2_model
import torchaudio.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 准备音频数据
SPEECH_FILE = 'audios/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
waveform, _ = torchaudio.load(SPEECH_FILE)
TRANSCRIPT = "i had that curiosity beside me at this moment".split()

# 2. 准备 wav2vec2 模型
# emissions 是音频每帧的特征，由特征提取算法，例如 MFCC 提取后，通过 Wav2Vec2 学习提取的特征
# 加载预先训练的 Wav2Vec2 模型
bundle = torchaudio.pipelines.MMS_FA
# star 指 <star> 特殊符号，当音频识别出的结果不在标签中时，使用 <star> 来代表该音频，类似文本处理中的 <unk>
model = bundle.get_model(with_star=False).to(device)
# 标签: 模型训练时能够识别的标签，也就是能够识别哪些字
LABELS = bundle.get_labels(star=None)
# 字典: 字典中包含标签及其对应的索引
DICTIONARY = bundle.get_dict(star=None)

# model = wav2vec2_model(
#     extractor_mode='layer_norm',
#     extractor_conv_layer_config=None,
#     extractor_conv_bias=False,
#     # 模型会根据此维度预测对应的概率个数，例如此处填1000，则代表词库中有1000个token对应模型输出
#     encoder_embed_dim=28,
#     encoder_projection_dropout=0.,
#     encoder_pos_conv_kernel=3,
#     # encoder_pos_conv_groups 必须整除 encoder_embed_dim
#     encoder_pos_conv_groups=7,
#     encoder_num_layers=3,
#     encoder_num_heads=4,
#     encoder_attention_dropout=0.,
#     encoder_ff_interm_features=1024,
#     encoder_ff_interm_dropout=0.,
#     encoder_dropout=0.,
#     encoder_layer_norm_first=True,
#     encoder_layer_drop=0.,
#     aux_num_out=None
# )

# 形状中的 28 代表 26 个英文字母 + 单引号 + 减号（blank占位符）
# LABELS = '-\'abcdefghijklmnopqrstuvwxyz'

# DICTIONARY = {LABELS[i]: i for i in range(len(LABELS))}

print(LABELS)
print(DICTIONARY)

print(model.__class__)

# 3. 使用 wav2vec2 模型提取音频特征 emissions
# (1, 169, 28)
emission, _ = model(waveform, None)

# 构造 token 索引列表
# 转换标签为数字
# tokens = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

# 用模型预测结果来生成 tokens
# (1, L, 28)
# emission.shape
tokens = []

for i in range(emission.shape[1]):
    tokens.append(torch.argmax(emission[0, i], dim=-1).item())

print(tokens)
tokens = []

# 上一个token
last_token = None
for i in range(emission.shape[1]):
    idx = torch.argmax(emission[0, i], dim=-1).item()
    if idx == 0:
        last_token = idx
        continue
    # 排除连续的相同 token
    elif idx == last_token:
        continue
    else:
        tokens.append(idx)
        last_token = idx

print(tokens)

# 对齐
# emission: wav2vec2 预测的结果
# tokens: 这段音频应该对应的文本 token 索引
def align(emission, tokens):
    # 函数 F.forced_align 接收的参数是包含批次信息的
    # 所以我们要对 tokens 展开一个维度充当批次
    target = torch.tensor([tokens], dtype=torch.int)
    # alignments: 音频对齐后的索引
    alignments, scores = F.forced_align(emission, target, blank=DICTIONARY['-'])
    # 取出第一个批次的数据
    alignments = alignments[0]
    scores = scores[0]
    # 将分数变成一个概率值
    scores = scores.exp()
    return alignments, scores


# 4. 使用 CTC 强制对齐 API 对齐音频 emissions 和文本 token
alignments, scores = align(emission, tokens)
print(alignments)

# 5. 使用令牌融合，将 CTC 强制对齐的内容做融合
# TokenSpan 的列表，TokenSpan 中包含一个时间段的开始帧和结束帧，以及对应的 token，还有 token 的得分
token_span = F.merge_tokens(alignments, scores)
print(token_span)

lengths = [len(word) for word in TRANSCRIPT]


# 单词对齐 将字母对齐成单词
# list_: TokenSpan 的列表
# lengths: 每个单词的长度
def unflatten(list_, lengths):
    words = []
    i = 0
    for l in lengths:
        words.append(list_[i:i + l])
        i += l
    return words


words_span = unflatten(token_span, lengths)
print(words_span)
words = [[LABELS[span.token] for span in word_span] for word_span in words_span]
words = [''.join(word) for word in words]
print(words)
