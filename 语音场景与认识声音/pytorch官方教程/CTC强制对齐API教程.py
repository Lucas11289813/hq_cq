# CTC 是 "Connectionist Temporal Classification"（连接主义时序分类）的缩写。CTC 是一种用于序列建模的技术，最初被用于语音识别，但后来也被应用于其他序列标记任务，如手写识别和自然语言处理。
# https://pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html

# 总结步骤:
# 1. 准备音频数据
# 2. 准备 wav2vec2 模型
# 3. 使用 wav2vec2 模型提取音频特征 emissions
# 4. 使用 CTC 强制对齐 API 对齐音频 emissions 和文本 token
# 5. 使用令牌融合，将 CTC 强制对齐的内容做融合

# 什么时候用对齐和融合？
# 当需要准确知道每个字的音频出现和结束的时间点，此时需要进行对齐，例如: 自动生成字幕等


import torch
import torchaudio
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

model = bundle.get_model(with_star=False).to(device)
# 标签: 模型训练时能够识别的标签，也就是能够识别哪些字
LABELS = bundle.get_labels(star=None)
# 字典: 字典中包含标签及其对应的索引
DICTIONARY = bundle.get_dict(star=None)

print(LABELS)
print(DICTIONARY)

print(model.__class__)
# 开启推理模式，关闭梯度和计算图的模式
with torch.inference_mode():
    # 3. 使用 wav2vec2 模型提取音频特征 emissions
    emission, _ = model(waveform.to(device))

# 形状中的 28 代表 26 个英文字母 + 单引号 + 减号（blank占位符）
print(emission.shape)


# 对齐
# emission: wav2vec2 预测的结果
# tokens: 这段音频应该对应的文本 token
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    # torch.Size([1, 37]) 目标token有37个
    print(targets.shape)
    # 形状张量(B, T, C)。其中 B 是批量大小，T 是输入长度，C 是字母表中的字符数（包括空格）
    # torch.Size([1, 169, 28]) 音频特征有 169 帧
    print(emission.shape)
    # 强制对齐
    # 需要和 CTC emisions 配合使用，也就是必须和 wav2vec2 模型预测结果一起使用
    alignments, scores = F.forced_align(emission, targets, blank=0)
    # torch.Size([1, 169]) 与 169 帧对齐后的 token
    print(alignments.shape)
    # torch.Size([1, 169]) 对齐评分
    # scores 代表的是强制对齐过程中每个音频帧与对应标签的对齐评分。这些评分是对齐模型对每个音频帧和对应标签之间对齐的确定程度的度量。
    print(scores.shape)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


# 转换标签为数字
tokens = [DICTIONARY[c] for word in TRANSCRIPT for c in word]
print(tokens)

# 4. 使用 CTC 强制对齐 API 对齐音频 emissions 和文本 token
alignments, scores = align(emission, tokens)
print(alignments.shape)

# 5. 使用令牌融合，将 CTC 强制对齐的内容做融合
# TokenSpan 的列表，TokenSpan 中包含一个时间段的开始帧和结束帧，以及对应的 token，还有 token 的得分
token_spans = F.merge_tokens(alignments, scores)
print(token_spans)

# 一行一行打印 tokenspan
print('Token\tStart\tEnd\tScore')
for ts in token_spans:
    print(f'{LABELS[ts.token]}\t[{ts.start:3d}, {ts.end:3d})\t{ts.score:.2f}')


# 单词对齐 将字母对齐成单词
# list_: TokenSpan 的列表
# lengths: 每个单词的长度
def unflatten(list_, lengths):
    # TokenSpan 长度严格等于文本总长
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    # 按照每个单词的长度取截取 token_spans 中的元素
    for l in lengths:
        ret.append(list_[i: i + l])
        i += l
    return ret


# 将 token_spans 按照单词进行分组
word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])
print(word_spans)

# 转换成 字符串
texts = [[LABELS[ts.token] for ts in ws] for ws in word_spans]
texts = [''.join(ts) for ts in texts]
print(texts)
