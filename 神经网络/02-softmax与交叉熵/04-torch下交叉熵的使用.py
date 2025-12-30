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

# 构建模型
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.Tanh(),
    # 在softmax中如果要做分类问题，需要将输出的神经元个数与分类数量大小相等
    nn.Linear(20, 2),
    # Softmax 需要加入属性dim
    nn.Softmax(dim=-1)
)
# --- 损失函数 ---
criterion = nn.CrossEntropyLoss()
# --- 优化器 ---
sgd = torch.optim.SGD(model.parameters(), lr=0.1)

# --- 训练 ----
epochs = 1000
for epoch in range(epochs):
    # 清空w，b 的梯度
    sgd.zero_grad()
    # 预测
    y_predict = model(x)
    # 交叉熵计算损失 注意原本的标签必须是整数
    # y_predict (80,2)
    # y (80,)
    loss = criterion(y_predict, y.long())
    # 反向传播更新梯度
    loss.backward()
    sgd.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")

y_predict = model(x)
# torch.argmax 某个方向最大值的下标  方向dim维度决定
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
y_test = model(test)
y_test = torch.argmax(y_test, dim=-1)
y_test = y_test.reshape(-1)
plt.subplot(122)
for i, (x, y) in enumerate(test):
    color = 'ro'
    if y_test[i] == 1:
        color = 'bo'
    plt.plot([x], [y], color)

plt.show()
