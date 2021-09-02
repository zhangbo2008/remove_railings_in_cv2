import cv2
import numpy as np
from numba import jit
import os



def gaussianFreqFilter(img, sigma, centX=None, centY=None):
    fft = np.fft.fft2(img)

    fft = np.fft.fftshift(fft)  # 图像傅里叶变换并移到图像中央位置
    cv2.imwrite('intermediate1.png', np.abs(fft))
    blur_img2 = np.fft.ifft2(fft)
    blur_img2 = np.abs(blur_img2)
    cv2.imwrite('intermediate3.png', blur_img2)
    cv2.imwrite( 'intermediate2.png',np.abs(fft))
    # 构造高斯核
    width, height = img.shape
    if centX is None and centY is None:
        centX = int(height/2)
        centY = int(width/2)
    Gauss_map1 = np.zeros((width, height))
    centX1 = centX
    centY1 = int(width / 2)
    for i in range(width):
        for j in range(height):
            dis = np.sqrt((i - centY1) ** 2 + (j - centX1) ** 2)
            Gauss_map1[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)  # 1.0- 表明是高通滤波器



    blur_fft = fft * Gauss_map1 #* Gauss_map2 * Gauss_map3 * Gauss_map4  # 这里面是hadmard积
    blur_img = np.fft.ifft2(blur_fft)
    blur_img = np.abs(blur_img)+255/2

    return blur_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    path='tmp.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('aaa.png',img)



    # img = cv2.resize(img,(2048,2048))
    # img_new = gaussianFreqFilter(img, 20.2, )
    # cv2.imwrite('out.png',img_new )
    # cv2.imwrite('out.png', np.uint16(img_new))
        # except:


        #     print(file)

    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread('tmp.png')

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite('dst.png',dst)
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)

    plt.savefig('out.png')


    print('done')
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    cv2.imwrite('box.png', box)





    #
    #
    #
    #
    #
    #
    #
    # Gauss_map2 = np.zeros((width, height))
    # centX2 = height - centX1
    # centY2 = int(width / 2)
    # for i in range(width):
    #     for j in range(height):
    #         dis = np.sqrt((i - centY2) ** 2 + (j - centX2) ** 2)
    #         Gauss_map2[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)
    #
    # Gauss_map3 = np.zeros((width, height))
    # centX3 = int(height / 2)
    # centY3 = centY
    # for i in range(width):
    #     for j in range(height):
    #         dis = np.sqrt((i - centY3) ** 2 + (j - centX3) ** 2)
    #         Gauss_map3[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)
    #
    # Gauss_map4 = np.zeros((width, height))
    # centX4 = int(height / 2)
    # centY4 = width - centY3
    #
    # for i in range(width):
    #     for j in range(height):
    #         dis = np.sqrt((i - centY4) ** 2 + (j - centX4) ** 2)
    #         Gauss_map4[i, j] = 1.0 - np.exp(-0.5 * dis / sigma)