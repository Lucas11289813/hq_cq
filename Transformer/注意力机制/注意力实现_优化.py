# 优化方法:
# 1. 点积注意力
# 2. 缩放点击注意力
# 具体步骤
# 1. Q K 的运算使用矩阵运算，不需要用循环
# 2. 论文简化了余弦相似度算法，相似度得分的 alpha = Q · K.T / dK
# 此处，底数也可以用 sqrt(dK) 替代 dK
# dK: 代表 K 的维度


import torch
import torch.nn as nn

torch.manual_seed(100)

# 词嵌入维度
embedding_dim = 100

text = 'how are you ?'
words = text.split(' ')

voc = torch.load('voc/vocab.voc')

# 转换索引
idx = torch.tensor(voc.lookup_indices(words))
# 使用嵌入层将单词变成词向量
embedding = nn.Embedding(len(voc), embedding_dim)
# 词向量 (4, 100)
word_vector = embedding(idx)

# 定义 QKV 矩阵的维度
q_dim = 100
k_dim = v_dim = 100

# 定义 QKV 权重
Wq = torch.randn(embedding_dim, q_dim)
Wk = torch.randn(embedding_dim, k_dim)
Wv = torch.randn(embedding_dim, v_dim)

# 求 QKV 矩阵
Q = word_vector @ Wq
K = word_vector @ Wk
V = word_vector @ Wv

# 使用简化版的公式求点积缩放注意力权重
weights = nn.functional.softmax(Q @ K.T / k_dim, dim=1)
# 计算最终的注意力
attention = weights @ V
print(attention)
print(attention.shape)
