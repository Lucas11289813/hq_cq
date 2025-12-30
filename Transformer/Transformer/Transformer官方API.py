from collections import OrderedDict

import torch
from torch import nn
from torchtext.vocab import vocab

# api: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer


# 注意: 官方API 不包含 词嵌入和位置编码部分
# src 长度可以和 tgt 长度不一致
# tgt 的长度决定了 Transformer 输出的序列长度

# d_model: 词嵌入维度 embed_dim
# nhead: 几个头
# num_encoder_layers: 编码器循环的次数
# num_decoder_layers: 解码器循环的次数
# dim_feedforward: 前馈神经网络中间层的维度
# dropout: 全连接后的正则化
# activation: 激活函数
# custom_encoder: 自定义编码器
# custom_decoder: 自定义解码器
# layer_norm_eps: 归一化时，分母为了防止为零的一个极小参数 epsilon
# batch_first: 是否批次在前面
# norm_first: 是否在注意力执行前先进行归一化
# bias: 线性层和归一化层是否使用偏置
model = nn.Transformer(
    d_model=100,
    nhead=2,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1,
    activation='relu',
    custom_encoder=None,
    custom_decoder=None,
    layer_norm_eps=1e-8,
    batch_first=True,
    norm_first=True,
    bias=False
)

# 构造输入
orderedDict = OrderedDict({
    'how': 500,
    'are': 500,
    'you': 500,
    '?': 500
})

voc = vocab(orderedDict, min_freq=1, specials=['<pad>', '<unk>', '<sos>', '<eos>'], special_first=True)
voc.set_default_index(voc['<unk>'])

src = 'how are you <pad> <pad> <pad>'.split(' ')
tgt = '<sos> how are you ? <eos>'.split(' ')

src_idx = torch.tensor(voc.lookup_indices(src)).unsqueeze(0)
tgt_idx = torch.tensor(voc.lookup_indices(tgt)).unsqueeze(0)

src_key_padding_mask = torch.zeros_like(src_idx, dtype=torch.float)
src_key_padding_mask[src_idx == voc['<pad>']] = float('-inf')

tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(src))

embedding = nn.Embedding(len(voc), 100)

# src: 输入序列
# tgt: 目标序列 输出序列
# src_mask: 输入序列的注意力掩码
# tgt_mask: 目标序列的注意力掩码
# memory_mask: 编码器输出结果的注意力掩码
# src_key_padding_mask: 输入序列的 padding 占位符的掩码
# tgt_key_padding_mask: 目标序列的 padding 占位符的掩码
# memory_key_padding_mask: memory 的 padding 占位符的掩码
# src_is_causal: src_mask 是否是因果注意力掩码
# tgt_is_causal: tgt_mask 是否是因果注意力掩码
# memory_is_causal: memory_mask 是否是因果注意力掩码
y = model(
    src=embedding(src_idx),
    tgt=embedding(tgt_idx),
    src_key_padding_mask=src_key_padding_mask,
    tgt_mask=tgt_mask,
    tgt_is_causal=True
)

print(y)
print(y.shape)
