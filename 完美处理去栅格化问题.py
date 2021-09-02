# writer:wojianxinygcl@163.com
#=============非常奇怪的一点, cv2保存傅里叶图片有黑图的bug, 用plt存就没问题了!!!!!!!!!!!\\\\\\


'''
原理:
https://baijiahao.baidu.com/s?id=1671643511992284379&wfr=spider&for=pc



'''


# date  : 2020.3.30

import cv2 as cv

import numpy as np

from matplotlib import pyplot as plt

#读取图像

img = cv.imread('tmp.png', 0)

#傅里叶变换

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f) # 把左上角的顶点,平移到中心.为了观看效果.

# fshift = np.uint8(fshift)
# #quzao
# import cv2
# fshift=cv2.fastNlMeansDenoisingColored(fshift, None, 10, 10, 7, 21)

if 1:
    sigma=0.0001
    width, height = img.shape
    centX = int(height / 2)
    centY = int(width / 2)
    Gauss_map1 = np.ones((width, height))
    # Gauss_map1[wi]
    canshu=0.01
    Gauss_map1[0:width//2-int(width*canshu),int(height)//2]=0
    Gauss_map1[width//2+int(width*canshu):,int(height)//2]=0
    plt.close()
    plt.imshow(Gauss_map1, 'gray')
    plt.savefig('3.png')
    fshift=fshift*Gauss_map1
    plt.close()
    plt.imshow(np.log(np.abs(fshift)) ) # res 只是为了观看., 'gray')
    plt.savefig('4.png')
    #查看fshift










#傅里叶逆变换

ishift = np.fft.ifftshift(fshift)

iimg = np.fft.ifft2(ishift)

iimg = np.abs(iimg)

plt.close()
plt.imshow(iimg)  # res 只是为了观看., 'gray')
plt.savefig('5.png')  # 最终去除掉栅栏的图片纯成5.png


