import torch
import torch.nn as nn


class NumberClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 声明卷积
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')

        # 全连接
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 全连接最后层的输出应该等于分类的数量

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.flatten = nn.Flatten(start_dim=1)

    # 输入图片的形状 (N, 1, 28, 28)
    # N: 批次数
    def forward(self, x):
        # 特征提取: 利用卷积和池化等操作提取特征
        x = self.conv1(x)
        # 激活
        x = self.relu(x)
        # N x 32 x 28 x 28
        x = self.pool(x)
        # N x 32 x 14 x 14
        x = self.conv2(x)
        # N x 64 x 14 x 14
        x = self.relu(x)
        x = self.pool(x)
        # N x 64 x 7 x 7

        # 展平
        x = self.flatten(x)
        # N x 3136

        # 分类: 利用全连接进行分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc3(x)
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    model = NumberClassifier()
    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    print(y.shape)
