"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: draw.py.py
@time: 2021/8/7 18:47

"""
import numpy as np
import cv2
def draw_fg(image,labels,anchors):
    ###############################################
    img_fg = image.copy()
    index = np.where(labels == 1)[0]
    fg_auchors = anchors[index]
    for item in fg_auchors:
        img_fg = cv2.rectangle(img_fg, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 0))
    cv2.imwrite('./result_fg.jpg', img_fg)
    #####################################################

def draw_bg(image,labels,anchors):
    ###############################################
    img_bg = image.copy()
    index = np.where(labels == 0)[0]
    bg_auchors = anchors[index]
    for item in bg_auchors:
        img_bg = cv2.rectangle(img_bg, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (255, 0, 0))
    cv2.imwrite('./result_bg.jpg', img_bg)
    #####################################################

def draw_class(image,class_labels,anchors):
    img_class = image.copy()
    color = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 0, 255), 3: (0, 255, 0), 4: (255, 0, 0)
        , 5: (0, 0, 0), 6: (255, 255, 0), 7: (125, 125, 0)}

    for i in range(int(class_labels.max()) + 1):
        index = np.where(class_labels == i)[0]
        class_auchors = anchors[index]
        for item in class_auchors:
            try:
                img_class = cv2.rectangle(img_class, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color[i])
            except:
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                img_class = cv2.rectangle(img_class, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color)
    cv2.imwrite('./result_class.jpg', img_class)
    #####################################################