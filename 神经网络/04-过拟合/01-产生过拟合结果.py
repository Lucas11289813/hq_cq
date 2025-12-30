import torch
import matplotlib.pyplot as plt
from torch import nn

x = torch.linspace(0, 1, 10)
y = 3 * x
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.plot(x, y, 'g^--', label="真实预测")
y += torch.normal(0, 0.3, size=y.shape)

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
criterion = nn.MSELoss()
# weight_decay 加入正则化的惩罚机制，介于[0,1]
sgd = torch.optim.SGD(model.parameters(), lr=0.05)
epochs = 5000
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

for epoch in range(epochs):
    sgd.zero_grad()
    y_predict = model(x)
    loss = criterion(y_predict, y)
    print(loss.item())
    loss.backward()
    sgd.step()

plt.plot(x, y, 'ro', label="原图")
y_predict = model(x)
plt.plot(x.detach().numpy(), y_predict.detach().numpy(), 'b--', label="训练")
plt.legend()
plt.show()
