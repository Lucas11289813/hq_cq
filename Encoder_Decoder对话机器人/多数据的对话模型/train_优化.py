import torch.optim
from torch import nn

from dataset import ChatDataset
from torch.utils.data import DataLoader, random_split
from model import ChatBot
from utils import save, load

# 超参数
EPOCH = 1
batch_size = 50
# lr = 1e-2

# 数据集
ds = ChatDataset()
# dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

# 模型
model, meta = load(ChatBot, 'weights')
lr = 1e-2 if len(meta['lr']) == 0 else meta['lr'][-1]
total_epoch = 0 if len(meta['epoch']) == 0 else meta['epoch'][-1]

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 学习率调度器，用于自动优化学习率
# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,  # 存储和使用学习率的优化器
    mode='min',  # 最大优化还是最小优化，监视学习率下降时，则是最小优化问题，输入 min，否则输入 max
    factor=0.1,  # 每次优化后学习率下降的倍数
    patience=3,  # 在多少个epoch内，如果优化器没有优化，则学习率下降
    threshold=0.0001,  # 优化器优化的阈值，当损失值变化量低于阈值，则认为学习率需要调整了
    cooldown=1,  # 学习率下降后，经过多少个epoch，再重新监视学习率下降
    min_lr=1e-6  # 学习率的最小值
)

# 损失函数
# ignore_index: 忽略指定索引对损失的影响
loss_fn = nn.NLLLoss(ignore_index=0)


# 有效损失函数
# outputs: 模型预测结果
# labels: 标签
def valid_loss_fn(outputs, labels):
    # 寻找符合条件的索引
    _, idx = torch.where(labels == 102)
    # 有效长度
    valid_lens = idx + 1

    # 存储符合条件的 y 和 lab
    yy = []
    labs = []

    for i in range(valid_lens.shape[0]):
        lab = labels[i]
        lab = lab[:valid_lens[i]]
        y = outputs[i]
        y = y[:valid_lens[i]]
        yy.append(y)
        labs.append(lab)

    y = torch.concat(yy, dim=0)
    lab = torch.concat(labs, dim=0)

    return loss_fn(y, lab)


def train(ds):
    print('train start')

    total_loss = 0.
    count = 0

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for i, (inputs, labels) in enumerate(dl):
        src, src_key_padding_mask, tgt, tgt_key_padding_mask = inputs

        optimizer.zero_grad()
        # 前向传播
        outputs = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
        labels = labels.reshape(-1)
        # 计算损失
        # loss = valid_loss_fn(outputs, labels)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        count += 1
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()
        if (i + 1) % 2 == 0:
            print(f'batch: {i + 1}; loss: {total_loss / count}')

    avg_loss = total_loss / count
    print(f'train over; loss: {avg_loss}')
    return avg_loss


def valid(ds):
    print('val start')

    total_loss = 0.
    count = 0

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    # 禁用梯度追踪
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dl):
            src, src_key_padding_mask, tgt, tgt_key_padding_mask = inputs

            optimizer.zero_grad()
            # 前向传播
            outputs = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
            outputs = outputs.reshape(outputs.shape[0] * outputs.shape[1], -1)
            labels = labels.reshape(-1)
            # 计算损失
            # loss = valid_loss_fn(outputs, labels)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            count += 1
            if (i + 1) % 2 == 0:
                print(f'batch: {i + 1}; loss: {total_loss / count}')

    avg_loss = total_loss / count
    print(f'val over; loss: {avg_loss}')
    return avg_loss


# 训练集长度
train_len = round(len(ds) * 0.8)
# 验证集长度
val_len = len(ds) - train_len

total_loss = 0.

for epoch in range(EPOCH):
    print(f'epoch: [{epoch + 1}/{EPOCH}]')

    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train(train_ds)
    loss = valid(val_ds)
    total_loss += loss

    scheduler.step(loss)

total_epoch += EPOCH

save(model, 'weights', total_epoch, total_loss / EPOCH, optimizer.param_groups[0]['lr'])
