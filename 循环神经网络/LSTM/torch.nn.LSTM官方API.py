# torch.nn.LSTM 的 API 和 torch.nn.RNN 基本相同
# 官方文档: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# 和 RNN 一样，LSTM 同样拥有官方的 LSTMCell，我们可以通过 LSTMCell 创建 一对一 一对多 多对一 多对多 结构的 GRU

import torch
import torch.nn as nn


# 多对一 LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)

    def forward(self, x, h=None, c=None):
        L, N, input_size = x.shape

        # 初始化长短期记忆
        if h is None:
            h = torch.zeros(N, self.hidden_size)
        if c is None:
            c = torch.zeros(N, self.hidden_size)

        # 循环
        for i in range(L):
            # 更新长短期记忆
            h, c = self.cell(x[i], (h, c))

        # 循环结束后，输出
        y = self.fc_out(h)
        return y, h, c


if __name__ == '__main__':
    # model = LSTM(2, 10)
    # x = torch.randn(3, 5, 2)
    # y, h, c = model(x)
    # print(y.shape, h.shape, c.shape)

    # 使用官方 LSTM
    # proj_size: 映射输出的长度，相当于全连接的输出
    # model = nn.LSTM(2, 10, batch_first=True, bidirectional=False, proj_size=2)
    model = nn.LSTM(2, 10, batch_first=True, bidirectional=False)
    x = torch.randn(5, 3, 2)
    y, (h, c) = model(x)
    print(y.shape, h.shape, c.shape)
    print(nn.Linear(10, 2)(y).shape)
