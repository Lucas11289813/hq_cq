import torch.optim
from torch import nn

from dataset import ChatDataset
from torch.utils.data import DataLoader
from model import ChatBot

# 超参数
EPOCH = 100
batch_size = 50
lr = 1e-2

# 数据集
ds = ChatDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model = ChatBot()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# 损失函数
loss_fn = nn.NLLLoss()

total_loss = 0.
count = 0

model.train()
for epoch in range(EPOCH):
    print(f'epoch: [{epoch + 1}/{EPOCH}]')

    for i, (inputs, labels) in enumerate(dl):
        src, src_key_padding_mask, tgt, tgt_key_padding_mask = inputs

        optimizer.zero_grad()
        # 前向传播
        outputs = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
        labels = labels.reshape(-1)
        # 计算损失
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        count += 1
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()

        print(f'batch: {i + 1}; loss: {total_loss / count}')

print(f'train over; loss: {total_loss / count}')
torch.save(model.state_dict(), 'model.pt')
