"""
当训练数据稀缺时，我们甚至可能无法提供足够的数据来构成一个合适的验证集。
这个问题的一个流行的解决方案是采用K折交叉验证。
这里，原始训练数据被分成K个不重叠的子集。
然后执行K次模型训练和验证，每次在K−1个子集上进行训练，
并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。
最后，通过对K次实验的结果取平均来估计训练和验证误差。
"""
import torch
from torch import nn

x = torch.normal(0, 1, (100,))
y = x ** 2
y += torch.normal(0, 0.2, (100,))

# --- K折交叉验证 ---
# 将数据集的格式进行设置
K = 5
# 算出K折每个数据集的大小
batch_size = len(x) // 5
tmp1 = []
tmp2 = []
# 将设置好的批次（batch_size）数据集设置出来
for i in range(K):
    # unsqueeze 在对应的位置上增加维度，-1 最后一个维度增加
    # squeeze 去除对应位置的维度
    tmp1.append(torch.unsqueeze(x[i * batch_size:(i + 1) * batch_size], -1))
    tmp2.append(torch.unsqueeze(y[i * batch_size:(i + 1) * batch_size], -1))

# 注意：
# 输入是两个维度：（批次，特征数量）
# 输入是多个维度：（批次，任意维度，特征数量）
x = torch.concatenate(tmp1, dim=-1)
x = x.reshape(K, -1, 1)
y = torch.concatenate(tmp2, dim=-1)
y = y.reshape(K, -1, 1)

# --- K折交叉验证 训练 ---
model = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)
criterion = nn.MSELoss()
sgd = torch.optim.Adam(model.parameters(), lr=0.01)
# 训练K次
for epoch in range(K):
    # 对K - 1个的数据集进行训练
    feature = x[:K - 1, :, :]
    sgd.zero_grad()
    predict = model(feature)
    # 第K个集作为验证集，验证该集的准确性 acc（线性回归中没有acc，只有valid loss）
    valid_feature = x[K - 1, None, :, :]
    # 防止w进行参与...
    with torch.no_grad():
        valid_labels = model(valid_feature)
        valid_loss = criterion(valid_labels, y[K - 1, None, :, :])
    loss = criterion(predict, y[:K - 1, :, :])
    loss.backward()
    print(f"epoch {epoch + 1} / {K} -- loss:{loss.item():.4f} -- valid loss:{valid_loss.item():.4f}")
    sgd.step()
