import cv2
import numpy as np
from skimage.util import random_noise

class DataAugmentatonMore():
    def __init__(self, image):
        self.image = image

    def motion_blur(self, degree=5, angle=180):
        # degree建议：2 - 5
        # angle建议：0 - 360
        # 都为整数
        image = np.array(self.image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred_image = np.array(blurred, dtype=np.uint8)
        return blurred_image

    def gaussian_blur(self, k_size=7, sigmaX=0, sigmaY=0):
        # k_size越大越模糊，且为奇数，建议[1，3，5，7，9]
        blurred_image = cv2.GaussianBlur(self.image, ksize=(k_size, k_size),
                                         sigmaX=sigmaX, sigmaY=sigmaY)
        return blurred_image

    def Contrast_and_Brightness(self, alpha, beta=0):
        # alpha:调节亮度，越小越暗，越大越亮，等于1为原始亮度
        # 建议使用0.6-1.3
        blank = np.zeros(self.image.shape, self.image.dtype)
        # dst = alpha * img + beta * blank
        brighted_image = cv2.addWeighted(self.image, alpha, blank, 1 - alpha, beta)
        return brighted_image

    def Add_Padding(self, top, bottom, left, right, color):
        padded_image = cv2.copyMakeBorder(self.image, top, bottom,
                                          left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_image

    def Add_gaussian_noise(self, mode='gaussian'):
        ##mode : 'gaussian' ,'salt' , 'pepper '
        noise_image = random_noise(self.image, mode=mode)
        return noise_image
    def resize_blur(self, ratio):
        w,h= self.image.shape[:2]
        image = cv2.resize(self.image, dsize=None, fy=ratio, fx=ratio)
        image = cv2.resize(image, dsize=(h, self.image.shape[0]))
        return image
    def transRGB(self):
        _type = np.random.choice([cv2.COLOR_RGB2BGR,cv2.COLOR_BGR2RGB],1)[0]
        img = cv2.cvtColor(self.image,_type)
        return img


def DataAugment(image):
    image = np.array(image)
    if (np.random.choice([True, False], 1)[0]):
        dataAu = DataAugmentatonMore(image)
        index = np.random.randint(0,5)
        if (index == 0):
            degree = np.random.randint(2, 6)
            angle = np.random.randint(0, 360)
            image = dataAu.motion_blur(degree, angle)
        elif (index == 1):
            id = np.random.randint(0, 4)
            k_size = [1, 3, 5, 7,9]
            image = dataAu.gaussian_blur(k_size[id])
        elif (index == 2):
            alpha = np.random.uniform(0.6, 1.3)
            image = dataAu.Contrast_and_Brightness(alpha)
        elif (index == 3):
            ratio = np.random.uniform(0.35, 0.5)
            image = dataAu.resize_blur(ratio)
        elif(index==4):
            image = dataAu.transRGB()
        del dataAu
    return image