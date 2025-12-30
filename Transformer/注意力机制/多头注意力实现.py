import math

import torch
import torch.nn as nn

torch.manual_seed(100)

embedding_dim = 100

text = 'how are you ?'
words = text.split(' ')

voc = torch.load('voc/vocab.voc')

idx = torch.tensor(voc.lookup_indices(words))
embedding = nn.Embedding(len(voc), embedding_dim)
word_vector = embedding(idx)

# 定义有几个头
nhead = 2
# 每个头的输入维度
head_dim = embedding_dim // nhead
# 分头
result = torch.chunk(word_vector, nhead, dim=1)
# QKV 权重，此处为了方便，我们将 q_dim = k_dim = v_dim = head_dim
QKV_weights = torch.randn(nhead, 3, head_dim, head_dim)

# outputs = []
#
# # 分头求QKV矩阵和其注意力Z
# for i in range(nhead):
#     # 拆分 QKV 的权重
#     Wq, Wk, Wv = torch.chunk(QKV_weights[i], 3, dim=0)
#     Wq, Wk, Wv = Wq.squeeze(0), Wk.squeeze(0), Wv.squeeze(0)
#     # 求 QKV 矩阵
#     Q = result[i] @ Wq
#     K = result[i] @ Wk
#     V = result[i] @ Wv
#     # 求注意力
#     Z = nn.functional.softmax(Q @ K.T / math.sqrt(head_dim)) @ V
#     outputs.append(Z)
#
# print(outputs)
#
# # 拼接多头注意力的输出
# tmp = torch.cat(outputs, dim=1)
#
# # 全连接输出
# fc = nn.Linear(embedding_dim, embedding_dim)
# # 得到注意力
# attention = fc(tmp)
# print(attention)
# print(attention.shape)


# 通过矩阵运算来计算多头注意力
# result (2, 4, 50)
# QKV_weights (2, 3, head_dim=50, head_dim=50)
result = torch.stack(result)

# 求 QKV 矩阵
# 使用批量矩阵乘法来进行运算
# Q (2, 4, 50)
Q = torch.bmm(result, QKV_weights[:, 0])
K = torch.bmm(result, QKV_weights[:, 1])
V = torch.bmm(result, QKV_weights[:, 2])

# 计算注意力权重
weights = nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_dim), dim=2)
Z = torch.bmm(weights, V)
# 连接多头注意力
Z = Z.reshape(-1, head_dim * nhead)

# 全连接输出
fc = nn.Linear(embedding_dim, embedding_dim)
attention = fc(Z)
print(attention)
print(attention.shape)
