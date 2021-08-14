"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: ctpn.py
@time: 2020/4/5 9:28

"""
import paddle
import paddle.nn as nn
from models.vgg import vgg16
from models.vggmy import vgg16_bn
import paddle.nn.functional as F


class Im2col(nn.Layer):
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, kernel_sizes=self.kernel_size, paddings=self.padding, strides=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x

class CTPN_Model(nn.Layer):
    def __init__(self):
        super(CTPN_Model, self).__init__()
        # self.cnn = VGG16(pretrained=True)
        self.cnn = vgg16_bn(pretrained=True)
        # self.cnn = vgg16(pretrained=True, batch_norm=False).features

        self.rpn = Im2col([3,3],[1,1],[1,1])
        self.brnn = nn.GRU(512*3*3,128,direction='bidirectional')
        self.lstm_fc = nn.Linear(256,512)
#######################################################################################################       
        self.vertical_coordinate = nn.Conv2D(512, 4 * 10, 1)
        self.score = nn.Conv2D(512, 2 * 10, 1)

    def forward(self, x):

        _batch_size = x.shape[0]
        x = self.cnn(x)
        x = self.rpn(x)

        x = paddle.transpose(x,(0,2,3,1)) # channels last
        b = x.shape  # batch_size, h, w, c
        x = paddle.reshape(x,(b[0]*b[1], b[2], b[3]))

        x, _ = self.brnn(x)
        x = self.lstm_fc(x)

        # import pdb
        # pdb.set_trace()
        x = paddle.reshape(x,(_batch_size, -1, x.shape[1], 512))
        x = paddle.transpose(x,(0,3,1,2)) # channels first
        ############################
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        return score,vertical_pred
