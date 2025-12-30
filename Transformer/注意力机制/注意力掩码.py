# 注意力掩码
# 什么是注意力掩码？
# 掩码: 用于计算 QK 点积时，叠加一个值（源码中是在计算QK点积前 torch.nn.functional 5439行）
# 形象的理解就是在输入序列上蒙住了部分的序列成员，让其不参与运算
# 有什么用？
# 用于抑制部分 QK 点积的结果，让其不参与注意力运算，通俗的理解就是，屏蔽部分序列中的成员在注意力计算时的效果
# 被抑制的成员，其注意力权重接近 0，意为完全不重要
import math

import torch
import torch.nn as nn

torch.manual_seed(100)

embedding_dim = 100

# 获取词向量
text = 'how are you ?'
words = text.split(' ')

voc = torch.load('voc/vocab.voc')

idx = torch.tensor(voc.lookup_indices(words))
embedding = nn.Embedding(len(voc), embedding_dim)
word_vector = embedding(idx)

# 声明全连接的权重
QKV_weights = torch.randn(3, embedding_dim, embedding_dim)
Wq, Wk, Wv = torch.chunk(QKV_weights, 3, dim=0)
Wq, Wk, Wv = Wq.squeeze(0), Wk.squeeze(0), Wv.squeeze(0)

# 计算 QKV 矩阵
Q = word_vector @ Wq
K = word_vector @ Wk
V = word_vector @ Wv

# 计算相似度
scores = Q @ K.T / math.sqrt(embedding_dim)
print(scores)

# 定义掩码
# 0: 不掩盖对应位置的值
# float('-inf'): 掩盖对应位置的值
attn_mask = torch.tensor([
    [0, 0, 0, float('-inf')],
    [0, float('-inf'), 0, 0],
    [float('-inf'), 0, 0, 0],
    [0, 0, 0, float('-inf')],
])

# 叠加掩码
_scores = scores + attn_mask

# 计算概率分布
weights = nn.functional.softmax(_scores, dim=-1)
print(weights)

# 常见的一种注意力掩码是三角形的注意力掩码，API: https://pytorch.org/docs/stable/generated/torch.triu.html#torch-triu
# torch.triu: 通过对角线，将对角线下方的值归零，对角线是从左上到右下
# diagonal: 对角线的移动单位
attn_mask = torch.ones_like(scores)
tmp = torch.triu(attn_mask, diagonal=1)
# torch.Tensor.masked_fill 用指定的值填充掩码
attn_mask = tmp.masked_fill(tmp != 0, float('-inf'))
print(attn_mask)

# 沿着左上到右下的对角线，下方为 0，上方为 -inf 的掩码，我们称为 因果注意力掩码
# 注意: 对角线上也是 0
