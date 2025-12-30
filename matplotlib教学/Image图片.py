import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

fig, ax = plt.subplots()

img = Image.open('./Lenna.png')
image = np.array(img)
print(image)
print(image.shape)

# 显示图片
# ax.imshow(image)

# 显示对应颜色通道的数据
# plt.imshow(image[:, :, 0])
# plt.imshow(image[:, :, 1])
# plt.imshow(image[:, :, 2])

# 显示颜色通道的图片
# image[:, :, 1] = 0
# image[:, :, 2] = 0
# image[:, :, 0] = 0
# image[:, :, 2] = 0
# image[:, :, 0] = 0
# image[:, :, 1] = 0
# ax.imshow(image)


# 转灰度图
# for x in range(image.shape[0]):
#     for y in range(image.shape[1]):
#         r = image[x, y, 0]
#         g = image[x, y, 1]
#         b = image[x, y, 2]
#         gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
#         # 赋值灰度值
#         image[x, y, 0] = gray
#         image[x, y, 1] = gray
#         image[x, y, 2] = gray
# ax.imshow(image)

# 使用单一通道的灰度图，显示成热力图
# 预设的Colormap有: https://matplotlib.org/stable/gallery/color/colormap_reference.html
# imgplot = plt.imshow(image[:, :, 0], cmap="hot")
# 修改 cmap
# imgplot.set_cmap('nipy_spectral')


# 图片直方图
# 直方图基本都在灰度图上进行
# .ravel() 展平数组
# fc  ec   facecolor edgecolor 的缩写
# plt.hist(image[:, :, 0].ravel(), bins=range(256), fc='k', ec='k')

# plt.imshow(image[:, :, 0])
# 限制颜色范围，不足40的设置到40，大于160的设置为160，用于增强对比
# 40, 160 这两个值是根据直方图的内容选择的
# plt.imshow(image[:, :, 0], clim=(40, 160))

# plt.colorbar()


# 数组插值，用于模糊处理
# 加载图片
img = Image.open('./stinkbug.png')
# 先获得图片的缩略图
img.thumbnail((64, 64))
image = np.array(img)
print(image)
print(image.shape)

# 支持的值:
# 'none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'.
# 双线性插值
imgplot = plt.imshow(image, interpolation="bilinear")
# 三线性插值
# imgplot = plt.imshow(image, interpolation="bicubic")
# 抗锯齿
# imgplot = plt.imshow(image, interpolation="antialiased")

# plt.imshow(image)

plt.show()
