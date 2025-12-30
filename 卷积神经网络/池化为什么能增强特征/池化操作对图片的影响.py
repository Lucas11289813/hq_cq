import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

image = Image.open('../Lenna.jpg')

img = ToTensor()(image)
print(img.shape)

# 池化操作
# 参数为池化核的大小
# 结论: 池化核大小为 2，那么原图的宽高将缩小为原来的二分之一
pool = torch.nn.MaxPool2d(2)

img = pool(img)
print(img.shape)

img = pool(img)
print(img.shape)

# 将张量转换回图片
image = ToPILImage()(img)
image.show()
