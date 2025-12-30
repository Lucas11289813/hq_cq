import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 1, 20)
y = 3 * x + 2
y += torch.normal(0, 0.2, y.shape)
plt.plot(x, y, 'ro--')

# 随机拟定一个斜率参数,requires_grad=True允许求梯度（斜率）
w = torch.tensor(0.1, requires_grad=True)
# 随机拟定一个截距参数
b = torch.tensor(0.2)
b.requires_grad = True  # 允许求梯度
# 学习率
lr = 0.05

y_predict = w * x + b
# detach() 对torch.tensor的拷贝
# numpy() 将torch.tensor转化为numpy
line, = plt.plot(x.detach().numpy(), y_predict.detach().numpy(), 'b^--')

# 训练的次数
epochs = 100
# 定义循环进行训练
for epoch in range(epochs):
    # 通过拟定的w、b进行y值的预测
    y_predict = w * x + b
    # 更新预测线
    line.set_data(x.detach().numpy(), y_predict.detach().numpy())
    # 损失函数 MSELoss 均方差损失 (a - b)^2
    loss = torch.mean((y_predict - y) ** 2)
    loss.backward()
    with torch.no_grad():
        # 使用loss对w、b进行求导
        slope_w = w.grad  # w关于loss的导数
        slope_b = b.grad  # b关于loss的导数
        # 梯度下降
        w -= lr * slope_w  # 更新w·
        b -= lr * slope_b  # 更新b
        # 清空每一次的导数值
        w.grad.zero_()
        b.grad.zero_()

    # .item() 将tensor以值的方式进行输出
    print(f"loss : {loss.item()}")
    # 每次更新的间隔事件
    plt.pause(0.5)
