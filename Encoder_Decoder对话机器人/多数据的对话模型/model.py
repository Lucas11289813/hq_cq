import torch
import torch.nn as nn
from FlagEmbedding import FlagModel


class PositionEmbedding(nn.Module):
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


# 对话模型
class ChatBot(nn.Module):
    def __init__(self, d_model=512, voc_size=21128):
        super().__init__()
        self.embedding = FlagModel(r'D:\projects\py-projects\bge-small-zh', use_fp16=True)
        self.pe = PositionEmbedding(embed_dim=d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            layer_norm_eps=1e-8,
            batch_first=True,
            norm_first=True
        )
        # 输出全连接
        self.fc_out = nn.Linear(d_model, voc_size)
        self.log_softmax = nn.LogSoftmax(-1)

    # (N, L)
    def embed_tokens(self, tokens):
        N, L = tokens.shape
        tokens = tokens.reshape(-1)
        tokens = self.embedding.tokenizer.convert_ids_to_tokens(tokens.numpy())
        tokens = self.embedding.encode(tokens)
        tokens = torch.from_numpy(tokens).reshape(N, L, -1)
        return tokens

    # src: (N, L)
    # tgt: (N, L)
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # 词嵌入
        src = self.embed_tokens(src)
        tgt = self.embed_tokens(tgt)

        # 位置编码
        src = self.pe(src)
        # (N, L, 512)
        tgt = self.pe(tgt)

        # 因果注意力掩码
        attn_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])

        # 调用 Transformer
        # (N, L, 512)
        output = self.transformer(
            src, tgt,
            tgt_mask=attn_mask,
            tgt_is_causal=True,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # (N, L, 21128)
        output = self.fc_out(output)

        output = self.log_softmax(output)

        return output


if __name__ == '__main__':
    from dataset import ChatDataset

    ds = ChatDataset()
    (src, tgt), label = ds[0]

    model = ChatBot()

    y = model(src.unsqueeze(0), tgt.unsqueeze(0))
    print(y)
    print(y.shape)
