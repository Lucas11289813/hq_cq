from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST


# 数据分割器
# 用于自动分割数据集获取训练集和验证集
# 控制一定周期内使用相同的数据集，超过一定周期后，我们重新随机数据集
class DatasetSpliter:
    def __init__(self, ds, batch_size, threshold=10, train_ds_rate=0.8):
        # train_ds_rate: 训练集的比例
        self.ds = ds
        self.batch_size = batch_size
        self.ds_len = len(self.ds)
        self.train_len = int(self.ds_len * train_ds_rate)
        self.val_len = self.ds_len - self.train_len
        # 阈值
        self.threshold = threshold
        # 当前已经调用 get_ds 的次数
        self.count = 0
        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None
        # 初始化数据集
        self._random_split_ds()

    # 获取数据集
    def get_ds(self):
        # 判断当前获取数据集是第几次获取
        if self.count >= self.threshold:
            # 若大于了获取阈值，则重新随机分割数据集
            self.count -= self.threshold
            self._random_split_ds()
        # 统计调用 get_ds 的次数
        self.count += 1
        # 返回分割后的数据集
        return self.train_ds, self.val_ds, self.train_dl, self.val_dl

    # 随机数据集
    def _random_split_ds(self):
        self.train_ds, self.val_ds = random_split(self.ds, [self.train_len, self.val_len])
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    ds = MNIST(root='./data', train=True)
    ds_spliter = DatasetSpliter(ds, threshold=3)
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    print(train_ds[0])
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    print(train_ds[0])
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    print(train_ds[0])
    train_ds, val_ds, train_dl, val_dl = ds_spliter.get_ds()
    print(train_ds[0])
