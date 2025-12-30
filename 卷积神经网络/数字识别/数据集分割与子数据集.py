from torchvision.datasets import MNIST
from torch.utils.data import random_split, Subset

ds = MNIST(root='./data', train=True)

# 随机分割数据集，可以用来做随机交叉验证
ds_len = len(ds)
# 训练集长度和验证集长度
train_len = int(ds_len * 0.8)
val_len = ds_len - train_len
# 随机分割数据集
train_ds, val_ds = random_split(ds, [train_len, val_len])
print(ds_len)
print(len(train_ds))
print(len(val_ds))
print(train_ds[0])

# 子集
# 第二个参数: 子数据集抽取原数据集中哪些索引的数据
# ss = Subset(ds, [0, 1, 2])
ss = Subset(ds, range(30000))
print(len(ss))

# random_split 和 Subset 创建的数据集都是 Dataset 的子类，可以像 Dataset 一样使用
