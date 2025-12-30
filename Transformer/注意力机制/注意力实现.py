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


# alpha 相似度函数
def alpha(_Q, _K):
    return torch.dot(_Q, _K) / (torch.sqrt(torch.dot(_Q, _Q)) * torch.sqrt(torch.dot(_K, _K)))


# 相似度得分矩阵
scores = torch.empty(Q.shape[0], K.shape[0])

# 计算相似度得分
for i in range(Q.shape[0]):
    for j in range(K.shape[0]):
        # 计算相似度得分
        score = alpha(Q[i], K[j])
        scores[i, j] = score

# 求每行的概率分布
# weights: 注意力权重
weights = nn.functional.softmax(scores, dim=1)
print(weights)

# tmp = torch.empty(weights.shape[0], weights.shape[1], 100)
#
# for i in range(weights.shape[0]):
#     for j in range(weights.shape[1]):
#         result = weights[i, j] * V[j]
#         tmp[i, j] = result
#
# print(tmp)
#
# # 求和
# # 得到注意力池
# attention = torch.sum(tmp, dim=1)
# print(attention)
# print(attention.shape)


# 扩展 weights 矩阵
_weights = weights.unsqueeze(2).expand(-1, -1, 100)
print(_weights.shape)
# 扩展 V 矩阵
_V = V.unsqueeze(0).expand(4, -1, -1)
print(_V.shape)
# 权重相乘
tmp = _weights * _V
# 求和，得到注意力池
attention = tmp.sum(dim=1)
print(attention)
print(attention.shape)

print(weights @ V)

