from chainer import Chain
import chainer.functions as F
import chainer.links as L

from pydlshogi.common import *

ch = 192
fcl = 256


class RoopBlock(Chain):
    def __init__(self,n_in,n_out,stride=1):
        super(RoopBlock,self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = n_in, out_channels = n_out, ksize = 3, pad = 1)
            self.conv2 = L.Convolution2D(in_channels = n_out, out_channels = n_out, ksize = 3, pad = 1)
            self.conv3 = L.Convolution2D(in_channels = n_out, out_channels = n_in, ksize = 3, pad = 1)

    def __call__(self,x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h)+x)

        return h    
    
  
class PolicyValueResnet(Chain):
    def __init__(self, blocks = 5):
        super(PolicyValueResnet, self).__init__()
        self.blocks = blocks
        with self.init_scope():
            self.l1=L.Convolution2D(in_channels = 104, out_channels = ch, ksize = 3, pad = 1)
            links = [("root0",RoopBlock(ch,ch))]
            n_in = ch
            n_out = 256
            for index in range(1,5,1):
                links += [("root{}".format(index),RoopBlock(n_in,n_out))]
                n_in *= 1
                n_out *= 1
            
            for link in links:
                self.add_link(*link)
            self.forward = links    
            # policy network
            self.policy=L.Convolution2D(in_channels = n_in, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1, nobias = True)
            self.policy_bias=L.Bias(shape=(9*9*MOVE_DIRECTION_LABEL_NUM))
            # value network
            self.value1=L.Convolution2D(in_channels = n_in, out_channels = MOVE_DIRECTION_LABEL_NUM, ksize = 1)
            self.value2=L.Linear(9*9*MOVE_DIRECTION_LABEL_NUM, fcl)
            self.value3=L.Linear(fcl, 1)

    def __call__(self, x):
        x = F.relu(self.l1(x))
        for name,func in self.forward:
            x = func(x)
        # policy network
        u_policy = self.policy(x)
        u_policy = self.policy_bias(F.reshape(u_policy, (-1, 9*9*MOVE_DIRECTION_LABEL_NUM)))
        # value network
        h_value = F.relu(self.value1(x))
        h_value = F.relu(self.value2(h_value))
        u_value = self.value3(h_value)
        return u_policy, u_value
