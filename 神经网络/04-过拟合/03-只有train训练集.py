import torch

# train 训练集（训练误差 -->  只要将数据集分为 train valid）
# features 特征  labels 标签 比如：零件长度、重量 就是特征  标签是哪一类零件...
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
print(len(train_data))
print(len(valid_data))