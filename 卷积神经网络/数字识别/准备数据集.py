# pytorch 官方已经封装了一些经典数据集

from torchvision.datasets import MNIST  # 手写数字数据集
from torchvision.transforms import ToTensor

# 下载官方数据
# ds = MNIST(root='./data', download=True, train=True)

# tt = ToTensor()
# 转换输入数据
# 该函数由数据集自己去调用
# def transform(inp):
#     return tt(inp)

# ds = MNIST(root='./data', train=True, transform=transform)

# 输出转换完后的数据
ds = MNIST(root='./data', train=True, transform=ToTensor())

# 数据集长度
print(len(ds))

# 用索引获取数据
data = ds[0]
# 取出输入数据和标签
inp, label = data

print(inp)
print(type(inp))
print(label)
