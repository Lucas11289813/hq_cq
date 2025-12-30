# torch.nn.functional.scaled_dot_product_attention 此 API 用于计算点乘缩放注意力
# nn.MultiheadAttention 此 API 用于计算多头注意力
# nn.Transformer.generate_square_subsequent_mask 生成掩码

import torch
import torch.nn as nn

# def div(a, b):
#     assert b != 0, 'b 不能是0'
#     return a / b
#
# print(div(1, 0))
# exit()


torch.manual_seed(100)

embedding_dim = 100

# 获取词向量
text = '<pad> how are you ? <pad>'
words = text.split(' ')

voc = torch.load('voc/vocab.voc')

idx = torch.tensor(voc.lookup_indices(words))
embedding = nn.Embedding(len(voc), embedding_dim)
word_vector = embedding(idx)

nhead = 5

# 参数:
# embed_dim: 嵌入维度，如果是语言模型的注意力计算，则该参数指的是词嵌入的长度
# num_heads: 多头注意力的头数
# dropout: 避免过拟合的随机抑制概率
# bias: 线性映射时是否添加偏置
# add_bias_kv: 是否给线性映射后的kv再添加一层偏置，该偏置不会加到最终结果，而是相当于多了一个中间特征，再计算过程中会增加一个嵌入长度的数据，最后矩阵乘法时被融合掉
#               例如: 我传入四个词嵌入，形状为 (4, 100)，add_bias_kv 为 True 时，kv 会被计算成 (5, 100)，多出来的 (1, 100) 就是添加的偏置信息，最后通过矩阵相乘，不影响最终结果形状
# add_zero_attn: 和 add_bias_kv 类似，会添加一层全是 0 的数据到 kv 中
# kdim、vdim: kv 维度的长度，默认和 embed_dim 相同。基本上 qkv 长度都应该相同，在 encoder-decoder 模型中，嵌入模型通常共享权重，输出长度相同
# batch_first: 批次维度是否放到首位
mha = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=nhead,
    dropout=0.,
    bias=False,
    add_bias_kv=True,
    add_zero_attn=True,
    batch_first=True
)

# 创建自注意力掩码
attn_mask = torch.ones(word_vector.shape[0], word_vector.shape[0])
tmp = torch.triu(attn_mask, diagonal=1)
attn_mask = tmp.masked_fill(tmp != 0, float('-inf'))

# padding-mask
padding_mask = torch.zeros_like(idx, dtype=torch.float)
padding_mask[idx == 1] = float('-inf')

# (1, 4, 100)
word_vector = word_vector.unsqueeze(0)
attn_mask = attn_mask.unsqueeze(0).expand(nhead, -1, -1)
padding_mask = padding_mask.unsqueeze(0)

# 输入参数:
# query、key、value: QKV映射前的矩阵
# key_padding_mask: key 中对于填充占位符的掩码
# need_weights: 是否返回注意力权重
# attn_mask: QK 点积后的掩码
# average_attn_weights: 返回各头的平均权重，否则返回每个头的权重
# is_causal: 是否使用因果掩码 若设置了 is_causal 为 True 则必须提供掩码
# # torch.nn.functional 5288 行提示: 可以使用 nn.Transformer.generate_square_subsequent_mask 函数生成掩码
# 返回注意力池和注意力权重
attention, weights = mha(
    query=word_vector,
    key=word_vector,
    value=word_vector,
    key_padding_mask=padding_mask,
    need_weights=True,
    attn_mask=attn_mask,
    average_attn_weights=True,
    is_causal=True
)

print(attention.shape)
print(weights)
print(weights.shape)


# 获取因果注意力掩码
print(nn.Transformer.generate_square_subsequent_mask(6))