import torch
from torch import nn

# from torchvision.ops import MLP

import torch
from torch import nn


# 位置编码
class PositionEncoding(nn.Module):
    # embed_dim: 词嵌入的维度
    # max_len: 可接收序列的最大长度
    def __init__(self, embed_dim=100, max_len=1000):
        super().__init__()
        # self.P 位置编码
        self.P = torch.zeros(max_len, embed_dim)
        # 构造正余弦函数的输入值
        pos = torch.arange(max_len).reshape(-1, 1)
        _2i = torch.arange(0, embed_dim, 2, dtype=torch.float)
        # 正余弦函数的自变量
        self.X = pos / torch.pow(10000, _2i / embed_dim)
        # 计算正余弦结果
        sin_result = torch.sin(self.X)
        cos_result = torch.cos(self.X)
        # 赋值正弦结果到位置编码P中
        self.P[:, 0::2] = sin_result
        # 赋值余弦结果到位置编码P中
        self.P[:, 1::2] = cos_result

    # x: (L, embed_dim)
    # x: (N, L, embed_dim)
    def forward(self, x):
        N, L, embed_dim = x.shape
        # 取位置编码
        # (L, embed_dim)
        position_encode = self.P[:L]
        # 扩展 N 个批次的维度
        # (1, L, embed_dim)
        position_encode.unsqueeze(0).expand(N, -1, -1)
        # 叠加位置编码
        x = x + position_encode
        return x


# 前馈神经网络，用于分类
class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim=1024, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return x


# 编码器
class Encoder(nn.Module):
    # embed_dim: 词嵌入长度
    # nhead: 几个头
    def __init__(self, embed_dim, nhead, dropout=0.):
        super().__init__()
        # 多头注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        # 层归一化
        # Normalization: 规范化，将数据映射为 均值为 0，方差为 1 的数据
        # norm = (x - mean) / (std**2 + epsilon)
        # BatchNorm: 批次维度上求归一化
        # LayerNorm: 最后一个维度上求归一化
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim)

    # src: (N, L, embed_dim)
    # key_padding_mask: 输入序列 key 中包含的 padding 掩码
    # attn_mask: 注意力掩码
    def forward(self, src, key_padding_mask=None, attn_mask=None):
        # 恒等映射
        identity = src
        # 编码器自注意力
        attention, weights = self.mha(
            src, src, src,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        # 残差连接
        memory = attention + identity
        # 归一化
        memory = self.norm(memory)
        # 恒等映射
        identity = memory
        # 前馈
        memory = self.ffn(memory)
        memory = memory + identity
        memory = self.norm(memory)
        return memory


# 解码器
class Decoder(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.):
        super().__init__()
        self.ffn = FFN(embed_dim)
        self.self_mha = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.en_de_mha = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    # tgt: (N, L, embed_dim)
    # memory: (N, L, embed_dim)
    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        N, L, embed_dim = tgt.shape
        # 恒等映射
        identity = tgt
        # 因果注意力掩码的自注意力
        _attn_mask = nn.Transformer.generate_square_subsequent_mask(L)
        # 融合掩码
        attn_mask = _attn_mask if attn_mask is None else attn_mask + _attn_mask
        attention, weights = self.self_mha(
            tgt, tgt, tgt,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=True
        )
        # 残差连接
        attention = attention + identity
        # 归一化
        attention = self.norm(attention)
        # 恒等映射
        identity = attention
        # 编解码注意力
        attention, weights = self.en_de_mha(
            attention, memory, memory,
        )
        # 残差连接
        attention = attention + identity
        # 归一化
        attention = self.norm(attention)
        # 恒等映射
        identity = attention
        y = self.ffn(attention)
        y = y + identity
        y = self.norm(y)
        return y


class Transformer(nn.Module):
    # voc_size: 词库的长度
    # N: 循环调用编码器和解码器的次数
    def __init__(self, voc_size, embed_dim, nhead, N, padding_idx=0):
        super().__init__()
        self.N = N
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=embed_dim, padding_idx=0)
        self.pe = PositionEncoding(embed_dim)
        self.encoder = Encoder(embed_dim, nhead)
        self.decoder = Decoder(embed_dim, nhead)
        # 输出的全连接
        self.fc_out = nn.Linear(embed_dim, voc_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, src, tgt, src_key_padding_mask=None, src_attn_mask=None, tgt_key_padding_mask=None,
                tgt_attn_mask=None):
        # 词嵌入
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 位置编码
        src = self.pe(src)
        tgt = self.pe(tgt)

        # 编码器的执行部分
        # 第一次执行编码器
        memory = self.encoder(src, src_key_padding_mask, src_attn_mask)
        # 循环 N - 1 次
        for i in range(self.N - 1):
            memory = self.encoder(memory)

        # 解码器的执行部分
        y = self.decoder(tgt, memory, tgt_key_padding_mask, tgt_attn_mask)
        for i in range(self.N - 1):
            y = self.decoder(y, memory)

        # 输出层全连接
        y = self.fc_out(y)
        y = self.softmax(y)

        return y


if __name__ == '__main__':
    embedding = nn.Embedding(6, 100, padding_idx=0)
    text = 'how are you <pad> <pad>'.split()
    src_idx = [1, 2, 3, 0, 0]
    # 目标序列第一个值为 <sos>，此处 <sos> 索引为 5
    # src = embedding(torch.tensor(src_idx)).unsqueeze(0)
    src = torch.tensor(src_idx).unsqueeze(0)
    # <sos> how are you ?
    tgt_idx = [5, 1, 2, 3, 4]
    # tgt = embedding(torch.tensor(tgt_idx)).unsqueeze(0)
    tgt = torch.tensor(tgt_idx).unsqueeze(0)

    key_padding_mask = torch.zeros(len(src_idx))
    key_padding_mask[src_idx == 0] = float('-inf')
    key_padding_mask = key_padding_mask.unsqueeze(0)

    # encoder = Encoder(100, 2)
    # decoder = Decoder(100, 2)
    # # key_padding_mask = torch.tensor([float('-inf'), 0, 0, 0, 0, float('-inf')])
    # memory = encoder(src, key_padding_mask)
    # y = decoder(tgt, memory)

    model = Transformer(6, 100, 2, N=3, padding_idx=0)
    y = model(src, tgt, key_padding_mask)
    print(y)
