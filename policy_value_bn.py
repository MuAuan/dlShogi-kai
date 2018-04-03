from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch = 192
fcl = 256
class PolicyValueNetwork(Chain):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            self.bn1   = L.BatchNormalization(ch)
            self.l2=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn2   = L.BatchNormalization(ch)
            self.l3=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn3   = L.BatchNormalization(ch)
            self.l4=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn4   = L.BatchNormalization(ch)
            self.l5=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn5   = L.BatchNormalization(ch)
            self.l6=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn6   = L.BatchNormalization(ch)
            self.l7=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn7   = L.BatchNormalization(ch)
            self.l8=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn8   = L.BatchNormalization(ch)
            self.l9=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn9   = L.BatchNormalization(ch)
            self.l10=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn10   = L.BatchNormalization(ch)
            self.l11=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn11   = L.BatchNormalization(ch)
            self.l12=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.bn12   = L.BatchNormalization(ch)
             #policy network
            self.l13=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.l13_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
             #value network
            self.l13_v=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.l14_v=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.l15_v=L.Linear(fcl, 1)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h1 = F.relu(self.bn1(h1))
        h2 = F.relu(self.l2(h1))
        h2 = F.relu(self.bn2(h2))
        h3 = F.relu(self.l3(h2))
        h3 = F.relu(self.bn3(h3))
        h4 = F.relu(self.l4(h3))
        h4 = F.relu(self.bn4(h4))
        h5 = F.relu(self.l5(h4))
        h5 = F.relu(self.bn5(h5))
        h6 = F.relu(self.l6(h5))
        h6 = F.relu(self.bn6(h6))
        h7 = F.relu(self.l7(h6))
        h7 = F.relu(self.bn7(h7))
        h8 = F.relu(self.l8(h7))
        h8 = F.relu(self.bn8(h8))
        h9 = F.relu(self.l9(h8))
        h9 = F.relu(self.bn9(h9))
        h10 = F.relu(self.l10(h9))
        h10 = F.relu(self.bn10(h10))
        h11 = F.relu(self.l11(h10))
        h11 = F.relu(self.bn11(h11))
        h12 = F.relu(self.l12(h11))
        h12 = F.relu(self.bn12(h12))
        # policy network
        h13 = self.l13(h12)
        policy = self.l13_bias(F.reshape(h13, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        h13_v = F.relu(self.l13_v(h12))
        h14_v = F.relu(self.l14_v(h13_v))
        value = self.l15_v(h14_v)
        return policy, value
