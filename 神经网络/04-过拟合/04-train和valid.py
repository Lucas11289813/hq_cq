import torch
from torch import nn

# train 训练集 valid 验证集（训练误差  泛化误差）

x1 = torch.linspace(1, 10, 10)
x2 = torch.linspace(1, 10, 10)
x1, x2 = torch.meshgrid((x1, x2), indexing='ij')
x1, x2 = x1.reshape(-1, 1), x2.reshape(-1, 1)
y = torch.concatenate((
    torch.zeros(40, ),
    torch.ones(30, ),
    torch.ones(30, ) * 2
)).reshape(-1, 1)

data = torch.concatenate((x1, x2, y), dim=-1)
print(data.shape)

# --- 根据比例分割数据 ----
# 7 份为train数据集  3份为valid数据集
n = data.shape[0] // 10
# 打乱数据集的顺序
indices = torch.randperm(data.shape[0])
data = data[indices]
# 分割数据
train_data = data[:n * 7]
valid_data = data[n * 7:]

epochs = 10
for epoch in range(epochs):
    # 分类问题: (因为分类才能计算准确率)
    # train数据集主要用于每个epoch的训练（训练集的损失函数）
    # valid 数据集主要用于每个epoch的accuracy（验证集的准确率）
    # 损失loss会从接近1的位置降到接近0的位置，accuracy会从接近0的位置，升到接近1的位置
    # 回归问题：
    # train数据集主要用于每个epoch的训练（训练集的损失函数）
    # valid 数据集主要用于每个epoch的验证（验证集的损失函数）
    # 损失loss会从接近1的位置降到接近0的位置
    pass
