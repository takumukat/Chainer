# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import time


class CNNLSTM(chainer.Chain):
    def __init__(self, out_unit):
        super(CNNLSTM, self).__init__(
            # Convolution2D(in_channels,out_channels,ksize,stride,pad)
            # NStepLSTM(layer, in, out, dropout)
            conv1=L.Convolution2D(1, 32, 5, 1, 2),
            conv2=L.Convolution2D(32, 32, 5, 1, 2),
            conv3=L.Convolution2D(32, 64, 5, 1, 2),
            fc4=L.Linear(None, 64),
            lstm5=L.NStepLSTM(1, 64, 64, 0.5),
            fc6=L.Linear(64, out_unit),
        )

    def CNN(self, x):
        h1 = self.conv1(x)
        h2 = F.max_pooling_2d(h1, ksize=3, stride=2)
        h3 = F.relu(h2)
        h4 = self.conv2(h3)
        h5 = F.relu(h4)
        h6 = F.average_pooling_2d(h5, ksize=3, stride=2)
        h7 = self.conv3(h6)
        h8 = F.relu(h7)
        h9 = F.average_pooling_2d(h8, ksize=3, stride=2)
        h10 = self.fc4(h9)
        #h10 = F.dropout(self.fc4(h9), ratio=0.5)

        return h10


    def __call__(self, hx, cx, x):
        #             |-----シーケンス長-----|
        #x: Variable([[frame],[frame],...], [[frame],[frame],...], ...)
        #            |-------------------バッチサイズ---------------------|
        h10 = [self.CNN(item.reshape(item.shape[0], 1, item.shape[1], item.shape[2])) for item in x]
        hy, cy, h11 = self.lstm5(hx, cx, h10)
        y = [self.fc6(item) for item in h11]

        return y