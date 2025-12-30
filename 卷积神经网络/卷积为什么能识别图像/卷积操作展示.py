from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn as nn

toTensor = ToTensor()  # 转换成张量

# 将转换器包装成转换器队列
transformer = Compose([
    toTensor,
    Resize((10, 10), antialias=False)
])

image = Image.open('../Lenna.jpg')

img = transformer(image)

# 声明卷积
conv = nn.Conv2d(3, 30, 3,
                 stride=1,
                 # 填充可以些数字，数字代表在图片上下左右格填充对应像素
                 # same: 填充内容让卷积后的图片大小和输入相同
                 # padding=2
                 # padding='same',
                 # 分组，数字必须被输出通道数整除，也不许被输入通道整除
                 # groups=3,
                 # 偏置
                 bias=False,
                 # 膨胀卷积，卷积核中每个权重间产生一个间距 s，s = dilation - 1
                 dilation=2
                 )
r = conv(img)
print(r.shape)
