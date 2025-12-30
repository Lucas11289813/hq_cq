import torch
import matplotlib.pyplot as plt
from torch import nn


def train(model, input, output, criterion, optimizer, epochs):
    """
    训练函数
    """
    for epoch in range(epochs):
        # 清空w，b 的梯度
        optimizer.zero_grad()
        # 预测
        y_predict = model(input)
        # 交叉熵计算损失 注意原本的标签必须是整数
        loss = criterion(y_predict, output.long())
        # 反向传播更新梯度
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")


# 自定义的神经网络模型
class CustomNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, num_classes),
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        return self.softmax(self.fc2(x))


# 计算准确率
def accuracy(predict, y):
    acc = sum(predict == y) / y.shape[0]
    return acc
