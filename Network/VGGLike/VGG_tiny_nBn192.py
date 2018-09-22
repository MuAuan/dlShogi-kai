from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch=192
class PolicyNetwork(Chain):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(104, ch, 3, pad=1)
            self.conv1_2 = L.Convolution2D(ch, ch, 3, pad=1)
            
            self.conv2_1 = L.Convolution2D(ch, ch*2, 3, pad=1)
            self.conv2_2 = L.Convolution2D(ch*2, ch*2, 3, pad=1)
            
            self.conv3_1 = L.Convolution2D(ch*2, ch*4, 3, pad=1)
            self.conv3_2 = L.Convolution2D(ch*4, ch*4, 3, pad=1)
            self.conv3_3 = L.Convolution2D(ch*4, ch*4, 3, pad=1)
            self.conv3_4 = L.Convolution2D(ch*4, ch*4, 3, pad=1)
            self.l13=L.Convolution2D(in_channels = ch*4, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.l13_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        #h = F.max_pooling_2d(h, 2, 2)
        #h = F.dropout(h, ratio=0.25)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        #h = F.max_pooling_2d(h, 2, 2)
        #h = F.dropout(h, ratio=0.25)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        #h = F.max_pooling_2d(h, 2, 2)
        #h = F.dropout(h, ratio=0.25)
        h13 = self.l13(h)
        return self.l13_bias(F.reshape(h13, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))

