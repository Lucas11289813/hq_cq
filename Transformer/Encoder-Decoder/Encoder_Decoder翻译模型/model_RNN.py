import random

import torch
from torch import nn
import torch.nn.functional as F
from dataset import TranslateDataset


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    # src 形状 (L, input_size)
    def forward(self, src, h):
        y, h = self.gru(src, h)
        return h


# 解码器
class Decoder(nn.Module):
    # max_iter: 解码器最大迭代次数
    # break_value_idx: 跳出循环的字符索引
    def __init__(self, input_size, hidden_size, max_iter=10, break_value_idx=1, teacher_forcing_ratio=0.8):
        super().__init__()
        self.input_size = input_size
        self.max_iter = max_iter
        self.break_value_idx = break_value_idx
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)
        # 随机一个 0~1 的值，和 teacher_forcing_ratio 做比较
        self.use_teacher_forcing = random.random() < teacher_forcing_ratio
        print(f'use teacher forcing: {self.use_teacher_forcing}')
        self.fc_all_out = nn.Linear(input_size, input_size)

    # memory 是编码器的输出 形状 (N, hidden_size)
    # tgt 形状 (L, N, input_size)
    # 我 爱 你 们
    def forward(self, tgt, memory):
        # tgt (L, input_size)
        identity = tgt

        outputs = []

        # 解码器的第一个输入为 <SOS>
        inp = F.one_hot(torch.tensor([0]), self.input_size)[0].unsqueeze(0).to(torch.float).to(tgt.device)

        for i in range(self.max_iter):
            out, memory = self.gru(inp, memory)
            # 预测一个字
            y = self.fc_out(memory)
            y = self.softmax(y)

            outputs.append(y)

            # 求最大值索引
            idx = y.argmax(-1)

            # 训练模式中，若已经输出了目标序列的个数，则跳出循环
            if self.training and len(outputs) >= tgt.shape[0]:
                break
            elif not self.training and idx.item() == self.break_value_idx:
                # 若预测结果等于 <EOS> 则 跳出循环
                break

            # 判断下一轮输入是强制教育还是自由运行模式
            if self.use_teacher_forcing:
                # 取出目标序列中对应值作为下一轮输入
                inp = tgt[i].unsqueeze(0).to(torch.float)
            else:
                # one-hot 编码
                # 本轮输出作为下一轮的输入
                inp = F.one_hot(idx, self.input_size).to(torch.float)

        # y (L, input_size)
        y = torch.cat(outputs, dim=0)
        # 残差连接
        y = y + identity
        y = self.fc_all_out(y)
        return y, memory


# 编解码器模型
class EncoderDecoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, max_iter=10, break_value_idx=1, teacher_forcing_ratio=0.8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size, max_iter, break_value_idx, teacher_forcing_ratio)

    # src 形状 (L, input_size): 输入序列
    # tgt 形状 (L, input_size): 训练用的目标序列
    def forward(self, src, tgt, h=None):
        if h is None:
            h = torch.zeros(1, self.hidden_size, device=src.device)
        # 编码
        memory = self.encoder(src, h)
        # 解码
        y, h = self.decoder(tgt, memory)
        return y, h


if __name__ == '__main__':
    encoder = Encoder(9, 20)
    decoder = Decoder(9, 20, teacher_forcing_ratio=0.8)
    decoder.train()
    # decoder.eval()
    ds = TranslateDataset()
    src, tgt = ds[0]
    h = torch.zeros(1, 20)
    memory = encoder(src, h)
    # 目标序列编码
    _tgt = F.one_hot(tgt, 9)
    # y, h = decoder(_tgt, memory)
    # print(y.shape)
    # print(h.shape)

    model = EncoderDecoderModel(9, 20)
    y, h = model(src, _tgt)
    print(y.shape)
    print(h.shape)
