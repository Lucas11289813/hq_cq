import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TranslateDataset
from model_RNNCell import EncoderDecoderModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = 5000
batch_size = 128
lr = 0.01

# 数据集
ds = TranslateDataset()

# 模型
model = EncoderDecoderModel(9, 20)
model.to(device)

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = 0.
count = 0

# 训练循环
model.train()
for epoch in range(EPOCH):
    src, tgt = ds[0]
    _tgt = F.one_hot(tgt, 9)
    src, _tgt, tgt = src.to(device), _tgt.to(device), tgt.to(device)
    optim.zero_grad()
    y, h = model(src, _tgt)
    loss = loss_fn(y, tgt)
    total_loss += loss.item()
    count += 1
    loss.backward()
    # 检查梯度长度，查看是否梯度消失
    for n, p in model.named_parameters():
        print(p.grad.norm())
    optim.step()
    # 每十个批次打印数据
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]')
        print(f'loss: {total_loss / count}')

print('train over')

print(f'loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model_rnncell.pt')
