from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 输出转换完后的数据
ds = MNIST(root='./data', train=True, transform=ToTensor())

# shuffle: 打乱顺序
dl = DataLoader(ds, batch_size=100, shuffle=True)

# 循环迭代数据
for i, (inputs, labels) in enumerate(dl):
    print(inputs)
    print(inputs.shape)
    print(labels)
    break
