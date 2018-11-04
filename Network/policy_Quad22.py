from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch = 192

class Block(Chain):

    def __init__(self):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.conv2 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.conv3 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            self.conv4 = L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.conv3(h3)
        return F.relu(x+h4) #x+h2

class PolicyNetwork(Chain):
    def __init__(self, blocks = 5):
        super(PolicyNetwork, self).__init__()
        self.blocks = blocks
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            self.l2=L.Convolution2D(in_channels = ch, out_channels = ch, ksize = 3, pad = 1)
            for i in range(1, 5):
                self.add_link('b{}'.format(i), Block())
            # policy network
            self.policy=L.Convolution2D(in_channels = ch, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.policy_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        for i in range(1, 5):
            h = self['b{}'.format(i)](h)
        # policy network
        h_policy = self.policy(h)
        u_policy = self.policy_bias(F.reshape(h_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        
        return u_policy
