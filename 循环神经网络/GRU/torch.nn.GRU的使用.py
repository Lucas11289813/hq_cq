# torch.nn.GRU 的 API 和 torch.nn.RNN 基本相同
# 官方文档: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru
# 和 RNN 一样，GRU 同样拥有官方的 GRUCell，我们可以通过 GRUCell 创建 一对一 一对多 多对一 多对多 结构的 GRU
import torch
from torch import nn


class ManyToOne_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size, bias=True)
        self.fc_out = nn.Linear(hidden_size, input_size)

    # x 形状 (N, L, input_size)
    # h 形状 (N, hidden_size)
    def forward(self, x, h=None):
        N, L, input_size = x.shape
        if h is None:
            h = torch.zeros(N, self.hidden_size)

        for i in range(L):
            h = self.cell(x[:, i], h)

        torch.flip(x, dims=[1])

        # 因为多对一结构，所以让所有的序列成员都执行完 GRUCell 后，得到最终隐藏状态后，再输出
        y = self.fc_out(h)
        return y, h


if __name__ == '__main__':
    model = ManyToOne_GRU(2, 10)
    x = torch.randn(5, 3, 2)
    y, h = model(x)
    print(y.shape)
    print(h.shape)
    # 官方nn.GRU的使用
    gru = nn.GRU(2, 10, num_layers=3, bias=True, batch_first=True, dropout=0.4, bidirectional=True)
    x = torch.randn(8, 3, 2)
    # y 形状 (N=8, L=3, hidden_size=20)
    # h 形状 (D=2*num_layers=3, N=8, hidden_size=10)
    y, h = gru(x)
    print(y.shape)
    print(h.shape)
    y = nn.Linear(20, 2)(y)
    y, h = gru(y, h)
    print(y.shape)
    print(h.shape)
