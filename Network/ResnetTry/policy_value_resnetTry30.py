from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch = 192
fcl = 256

class Block(Chain):

    def __init__(self):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.conv2 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.conv3 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        return F.relu(x+h3) #x+h2

class PolicyValueResnet(Chain):
    def __init__(self, blocks = 10):
        super(PolicyValueResnet, self).__init__()
        self.blocks = blocks
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            self.l2=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            for i in range(1, 10):
                self.add_link('b{}'.format(i), Block())
            # policy network
            self.policy=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.policy_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            # value network
            self.value1=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.value2=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.value3=L.Linear(fcl, 1)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        for i in range(1, 10):
            h = self['b{}'.format(i)](h)
        # policy network
        h_policy = self.policy(h)
        u_policy = self.policy_bias(F.reshape(h_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        h_value = F.relu(self.value1(h))
        h_value = F.relu(self.value2(h_value))
        u_value = self.value3(h_value)
        return u_policy, u_value
