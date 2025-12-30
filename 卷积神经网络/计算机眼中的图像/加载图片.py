from PIL import Image
import numpy as np
import cv2

# # 打开一张图片
# image = Image.open('../Lenna.png')
# print(image)
#
# # 转换numpy数组
# img = np.array(image)
#
# # print(img)
# # print(img.shape)
#
# # image.show()
#
# # 分离颜色通道
# # img[:, :, 0]
# # # 其他颜色通道设为 0
# # img[:, :, 1] = 0
# # img[:, :, 2] = 0
#
# img[:, :, 0] = 0
# # img[:, :, 1]
# img[:, :, 2] = 0
#
# image = Image.fromarray(img)
# image.show()


img = cv2.imread('../Lenna.png')
print(img.shape)
img[:, :, 0] = 0
img[:, :, 1] = 0
cv2.imshow('Lenna', img)
cv2.waitKey(0)
