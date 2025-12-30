# 双向RNN是在单向RNN的基础上，再增加一个 RNNCell，然后将序列反向输入
# 最终 BiRNN 的输出和隐藏状态分别是:
# 输出: 将正反向输出在特征维度上进行拼接
# 隐藏状态: 将正反向隐藏状态进行 stack 堆叠
import torch
from torch import nn


# 双向RNN的优点是，让模型考虑序列的过去和未来的发展趋势。

# (L, input_size)
# x
#
# hey h
#
# forward = RNNCell
#
# (L, hidden_size)
# y1, h1
#
# h yeh
#
# backward = RNNCell
#
# (L, hidden_size)
# y2, h2
#
# (L, 2 * hidden_size)
# y3
#
# h1 (1, hidden_size)
# h2 (1, hidden_size)
# h3 (2, 1, hidden_size)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 前向rnn
        self.forward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 反向rnn
        self.backward_rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    # x 形状 (N, L, input_size)
    def forward(self, x, h=None):
        # 调用前向 rnn
        y1, h1 = self.forward_rnn(x, h)
        # 反转数据
        x = torch.flip(x, [1])
        # 调用反向 rnn
        y2, h2 = self.backward_rnn(x, h)
        # 拼接输出
        y = torch.cat([y1, y2], dim=2)
        # 堆叠隐藏状态
        h = torch.cat([h1, h2], dim=0)
        return y, h


if __name__ == '__main__':
    # 官方的双向RNN
    model = nn.RNN(2, 10, bidirectional=True, batch_first=True)
    # model = BiRNN(2, 10)
    x = torch.randn(5, 3, 2)
    y, h = model(x)
    print(y.shape, h.shape)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y.reshape(15, 20), torch.randint(0, 20, (15,)))
    loss.backward()
