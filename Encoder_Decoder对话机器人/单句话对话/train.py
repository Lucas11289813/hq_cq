import torch
from torch import nn

from dataset import ChatDataset
from model import ChatBot

# 超参数
EPOCH = 100
lr = 1e-3

# 数据集
ds = ChatDataset()

# 模型
model = ChatBot()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 损失函数
loss_fn = nn.NLLLoss()

(src, tgt), label = ds[0]
src, tgt, label = src.unsqueeze(0), tgt.unsqueeze(0), label.unsqueeze(0)

total_loss = 0.
count = 0

model.train()
for epoch in range(EPOCH):
    # 清空梯度
    optimizer.zero_grad()
    # 前向传播
    y = model(src, tgt)
    # 变形处理
    y = y.reshape(y.shape[0] * y.shape[1], -1)
    label = label.reshape(-1)
    # 求损失
    loss = loss_fn(y, label)
    total_loss += loss.item()
    count += 1
    # 反向传播
    loss.backward()
    # 优化模型
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'model.pt')
