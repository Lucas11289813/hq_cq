import torch
import matplotlib.pyplot as plt
from torch import nn

x = torch.linspace(0, 1, 20)
y = 3 * x + 2
y += torch.normal(0, 0.2, y.shape)
plt.plot(x, y, 'ro--')

# 构建模型  关键点：1个输入与1个输出
# nn.Sequential 将神经网络功能模型进行组装的容器（顺序 ：最上层是输入，最下层是输出）
# nn.Linear Linear是线性的意思，实际上就是y = w * x + b  神经网络的一个线性功能模块
model = nn.Sequential(
    # bias=True 在原本的线性公式中加上b，如果为False则不加, 默认情况下bias为True
    nn.Linear(1, 1, bias=True)
)

# 损失函数  MSELoss 均方差（最小二乘法）
criterion = nn.MSELoss()

# 查看models中的w和b
# for param in model.parameters():
#     print(param)

# 优化器  => w新 = w旧 - learning_rate * grad(斜率：二维空间、梯度：高维空间) => SGD
# optimizer => optim 优化器
# w => 从model中去取 => model.parameters() 取出model中的所有的w,b
sgd = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练的次数
epochs = 100
# 对x和y进行预处理
# pytorch在训练过程中，需要如何的参数的前两个维度必须是 （batch_size,in_features）
#                                                （批次数量,输入特征）
x = x.reshape(-1, 1)  # batch_size = 20
y = y.reshape(-1, 1)
# 定义循环进行训练
for epoch in range(epochs):
    # 每次运算，清空之前w和b的梯度
    sgd.zero_grad()
    # 通过模型之后的预测值
    y_predict = model(x)
    # 计算损失
    loss = criterion(y_predict,y)
    # 反向传播
    loss.backward()
    # 更新所有的w,b
    sgd.step()
    print(f"epoch {epoch + 1} / {epochs} -- loss: {loss.item():.2f}")

# 每次更新的间隔事件
# plt.pause(0.5)
