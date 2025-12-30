import torch
from torch import nn


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


if __name__ == '__main__':
    pe = PositionEncoding()
    x = torch.randn(6, 5, 100)
    x = pe(x)
    print(x)
