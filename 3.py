# writer:wojianxinygcl@163.com

# date  : 2020.3.30

import cv2 as cv

import numpy as np

from matplotlib import pyplot as plt

#读取图像

img = cv.imread('tmp.png', 0)

#傅里叶变换

f = np.fft.fft2(img)

fshift = f
sigma=300

# fshift[:,0]=fshift[:,1]
# fshift[0,:]=fshift[1,:]
# width, height = img.shape
# centX = int(height / 2)
# centY = int(width / 2)
# Gauss_map1 = np.zeros((width, height))
# centX1 = centX
# centY1 = int(width / 2)
# for i in range(width):
#     for j in range(height):
#         dis = np.sqrt((i - centY1) ** 2 + (j - centX1) ** 2)
#         Gauss_map1[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)  # 1.0- 表明是高通滤波器

res = np.log(np.abs(fshift))

#傅里叶逆变换

ishift = fshift

iimg = np.fft.ifft2(ishift)

iimg = np.abs(iimg)

#展示结果

plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')

plt.axis('off')

plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')

plt.axis('off')

plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')

plt.axis('off')

plt.savefig("fuliye.png")
plt.close()
plt.imshow(res,'gray')
plt.savefig('333333333333.png')
cv.imwrite('fuliyetupian.png',res)