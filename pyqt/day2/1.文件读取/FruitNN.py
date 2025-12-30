# 定义神经网络
# 定义类，继承torch.nn.Module
# 构造函数定义神经网络序列
# 前向传播函数调用序列
from torch import nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.seq = nn.Sequential(
            # 将原图3通道100*100   --->    n个10以内的特征图，展平、全连接  --->  3个结果
            # 第一次卷积，使用10个5*5卷积核  100-5+1 --> 96*96
            nn.Conv2d(3,10,5),
            # 归一化、激活函数
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 第一次池化,使用2*2窗口，步长2,  96 --> 48*48
            nn.MaxPool2d(2,2),
            # 第二次卷积，使用20个5*5卷积核  48-5+1 --> 44*44
            nn.Conv2d(10, 20, 5),
            # 归一化、激活函数
            nn.BatchNorm2d(20),
            nn.ReLU(),
            # 第二次池化，使用2*2窗口，步长2, 44 --> 22*22
            nn.MaxPool2d(2, 2),
            # 第三次卷积，使用30个5*5卷积核 22-5+1 --> 18*18
            nn.Conv2d(20, 30, 5),
            # 归一化、激活函数
            nn.BatchNorm2d(30),
            nn.ReLU(),
            # 第三次池化，使用2*2窗口，步长2, 18 --> 9*9
            nn.MaxPool2d(2, 2),
            # 正则化
            nn.Dropout(),
            # 展平
            nn.Flatten(),
            # 全连接 30*9*9 --> 300
            nn.Linear(30*9*9,300),
            # 全连接 300 --> 75
            nn.Linear(300,75),
            # 全连接 75 --> 25
            nn.Linear(75, 25),
            # 全连接 25 --> 3
            nn.Linear(25, 3)
        )

    def forward(self, x):
        return self.seq(x)
