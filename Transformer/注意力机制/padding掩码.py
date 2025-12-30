# 什么是 padding-mask？
# 语句中包含的 padding 占位符，在计算注意力掩码时，不重要，所以需要掩码频闭
# 这种掩码称为 padding-mask
import math

import torch
import torch.nn as nn

torch.manual_seed(100)

embedding_dim = 100

# 获取词向量
text = '<pad> how are you ? <pad>'
words = text.split(' ')

voc = torch.load('voc/vocab.voc')

idx = torch.tensor(voc.lookup_indices(words))
embedding = nn.Embedding(len(voc), embedding_dim, padding_idx=1)
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

# 声明一个因果注意力掩码
attn_mask = torch.ones_like(scores)
tmp = torch.triu(attn_mask, diagonal=1)
attn_mask = tmp.masked_fill(tmp != 0, float('-inf'))

# 声明 padding-mask
# 不是 <pad> 占位符的位置填 0，否则填 -inf
# padding_mask = torch.tensor([float('-inf'), 0, 0, 0, 0, float('-inf')])
padding_mask = torch.zeros_like(idx, dtype=torch.float)
padding_mask[idx == 1] = float('-inf')
print(padding_mask)

# 融合 attn_mask 和 padding_mask
# 融合规则: 将 padding_mask 复制成和 attn_mask 一样的形状，然后再叠加
padding_mask = padding_mask.unsqueeze(0).expand_as(attn_mask)
attn_mask = attn_mask + padding_mask
print(attn_mask)

# 相似度得分叠加注意力掩码
_scores = scores + attn_mask

# 求注意力权重
weights = nn.functional.softmax(_scores, dim=1)
print(weights)
