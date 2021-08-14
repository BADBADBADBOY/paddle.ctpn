import paddle
import paddle.nn as nn
from models.vgg import vgg16
from models.vggmy import vgg16_bn
import paddle.nn.functional as F

class BasicConv(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,weight_attr=nn.initializer.KaimingUniform())
        self.bn = nn.BatchNorm2D(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CTPN_Model(nn.Layer):
    def __init__(self):
        super(CTPN_Model, self).__init__()
        # self.cnn = VGG16(pretrained=True)
        self.cnn = vgg16_bn(pretrained=True)
        # self.cnn = vgg16(pretrained=True, batch_norm=False).features

        self.rpn = BasicConv(512, 512, 3,1,1,bn=False)
        self.brnn = nn.GRU(512,128,direction='bidirectional')
        self.lstm_fc = BasicConv(256, 512,1,1,relu=True, bn=False)

#######################################################################################################       
        self.vertical_coordinate = nn.Conv2D(512, 4 * 10, 1)
        self.score = nn.Conv2D(512, 2 * 10, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rpn(x)

        x1 = paddle.transpose(x,(0,2,3,1)) # channels last
        b = x1.shape  # batch_size, h, w, c
        x1 = paddle.reshape(x1,(b[0]*b[1], b[2], b[3]))

        x2, _ = self.brnn(x1)

        b = x.shape
        x2 = paddle.reshape(x2,(b[0], b[2], b[3], 256))
        x2 = paddle.transpose(x2,(0,3,1,2)) # channels first
        x2 = self.lstm_fc(x2)

        ############################
        vertical_pred = self.vertical_coordinate(x2)
        score = self.score(x2)
       
        return score,vertical_pred