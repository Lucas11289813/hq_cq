import torch
import matplotlib.pyplot as plt
from torch import nn

x = torch.linspace(-5, 5, 40)
y = torch.cos(x) * x
plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro-')
model = nn.Sequential(
    # 神经网络输入
    nn.Linear(1, 10, bias=True),  # y = w * x + b
    # 激活函数
    nn.Tanh(),  # y = tanh(x)  -1,1
    # 神经网络输出
    nn.Linear(10, 1, bias=True),  # y = w * x + b
)

criterion = nn.MSELoss()

sgd = torch.optim.SGD(model.parameters(), lr=0.05)

epochs = 4000
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# y_predict = model(x)
# line, = plt.plot(x.detach().numpy(), y_predict.detach().numpy(), 'b--')
for epoch in range(epochs):
    sgd.zero_grad()
    y_predict = model(x)
    # line.set_data(x.detach().numpy(), y_predict.detach().numpy())
    loss = criterion(y_predict, y)
    loss.backward()
    sgd.step()
    print(f"epoch {epoch + 1} / {epochs} -- loss: {loss.item():.2f}")
    # plt.pause(0.1)

y_predict = model(x)
plt.plot(x.detach().numpy(), y_predict.detach().numpy(), 'b--')
plt.show()

