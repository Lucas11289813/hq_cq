# Dataset 是pytorch定义的数据集类型
from torch.utils.data import Dataset


# 自定义数据集
class MyDataset(Dataset):
    # Dataset 需要实现 __init__ __len__ __getitem__ 三个方法
    # __init__ 初始化数据集
    # __len__ 返回数据集的长度
    # __getitem__ 返回一条数据和对应标签
    # transform: 数据预处理，通常转换成张量
    # target_transform: 标签预处理
    # transform 和 target_transform 都是函数
    def __init__(self, transform=None, target_transform=None):
        super().__init__()
        self.data = [('汽车', 0), ('飞机', 1), ('汽车', 0), ('轮船', 2)]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    # 获取一条数据和对应标签
    # 需要返回给模型的数据和标签
    def __getitem__(self, index):
        data = self.data[index]
        inp, label = data
        return inp, label
