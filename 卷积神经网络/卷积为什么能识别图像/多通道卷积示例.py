from PIL import Image
# 引入pytorch转换器
from torchvision.transforms import Grayscale, ToTensor, Resize, Compose
import torch.nn as nn

toTensor = ToTensor()  # 转换成张量
resize = Resize((200, 200), antialias=True)  # 重置图片大小

# 将转换器包装成转换器队列
transformer = Compose([
    toTensor,
    resize
])

# png 图片是 rgba 4通道
# jpg 图片是 rgb 3通道
image = Image.open('../Lenna.png')

img = transformer(image)

print(img)
print(img.shape)

# convolution
# 第一个参数: 输入通道数
# 第二个参数: 输出通道数==卷积核个数
# 第三个参数: 卷积核大小
# 返回卷积操作函数
conv = nn.Conv2d(4, 64, 3)
# 调用卷积操作
result = conv(img)
print(result)
print(result.shape)
