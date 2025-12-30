import torch
import matplotlib.pyplot as plt
from torch import nn

plt.subplot(121)
# 簇族A = 0
a = torch.normal(0.5, 0.4, (40, 2))
plt.plot(a[:, 0], a[:, 1], 'ro')
a_labels = torch.zeros((40, 1))
a = torch.concatenate((a, a_labels), dim=1)

# 簇族B = 1
b = torch.normal(1.5, 0.4, (40, 2))
plt.plot(b[:, 0], b[:, 1], 'bo')
b_labels = torch.ones((40, 1))
b = torch.concatenate((b, b_labels), dim=1)

data = torch.concatenate((a, b), dim=0)
# torch中打乱所有点的顺序
indices = torch.randperm(data.shape[0])
data = data[indices]
x = data[:, :2]  # (80,2)
y = data[:, 2]  # (80,)
y_copy = y.detach()


# 定义tanh激活函数
def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


# 定义softmax函数
def softmax(x):
    s = torch.sum(torch.exp(x), dim=1).reshape(-1, 1)
    s = torch.concatenate((s, s), dim=1)
    return torch.exp(x) / s


# 定义w,b
w = torch.tensor([[0.1, 0.1]], requires_grad=True)
b = torch.tensor([[0.1, 0.1]], requires_grad=True)

# 独热编码
y = nn.functional.one_hot(y.long())

# 进行模型训练
epochs = 100
for epoch in range(epochs):
    # softmax之后的结果
    y_predict = softmax(tanh(x * w + b))
    # 交叉熵损失
    loss = -torch.mean(y * torch.log(y_predict))
    print(f"loss: {loss}")
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
        w.grad.zero_()
        b.grad.zero_()

y_predict = softmax(tanh(x * w + b))
result = torch.argmax(y_predict, dim=-1)

# 准确率： 正确的样本在整体中的出现的概率
acc = sum(result == y_copy) / y_copy.shape[0]
print(acc)

# 测试样本
x1_test = torch.linspace(0, 2, 20)
x2_test = torch.linspace(0, 2, 20)
x1_test, x2_test = torch.meshgrid((x1_test, x2_test), indexing='ij')
x1_test = x1_test.reshape(-1, 1)
x2_test = x2_test.reshape(-1, 1)
test = torch.concatenate((x1_test, x2_test), dim=1)
y_test = softmax(tanh(test * w + b))
y_test = torch.argmax(y_test, dim=-1)
y_test = y_test.reshape(-1)
plt.subplot(122)
for i, (x, y) in enumerate(test):
    color = 'ro'
    if y_test[i] == 1:
        color = 'bo'
    plt.plot([x], [y], color)

plt.show()
