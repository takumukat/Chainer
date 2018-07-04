# -*- coding: utf-8 -*-
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class RNN_model(chainer.Chain):
    def __init__(self):
        super(RNN_model, self).__init__(
        #1-10-10-1のsin関数のFFNN学習にLSTMを入れたもの
            l1 = L.Linear(2,100),
            l2 = L.NStepLSTM(1, 100, 100, 0.5), #LSTM
            l3 = L.Linear(100,5),
        )
    
    #回帰問題なので二乗平均誤差
    def __call__(self, hx, cx, x, t):
        loss = None
        acc = []
        for f, t in zip(self.fwd(x, hx, cx), t):
            if loss is not None:
                loss += F.softmax_cross_entropy(f, t)
            else:
                loss = F.softmax_cross_entropy(f, t)
            acc.append( F.accuracy(f, t) )
        
        return loss, acc
    
    #順伝播の計算
    def fwd(self, x, hx, cx):
        h1 = [self.l1(item) for item in x]
        hy, cy, h2 = self.l2(hx, cx, h1)
        h3 = [self.l3(item) for item in h2]
        return h3