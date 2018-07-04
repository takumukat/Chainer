import chainer
import chainer.links as L
import chainer.functions as F

#---------------------------------------rightleft result------------------------------------------
#epoch: 50  train_loss: 0.0000  train_accuracy: 1.0000   test_loss: 0.0021   test_accuracy: 1.0000
#-------------------------------------------------------------------------------------------------

class MyCNN(chainer.Chain):
    def __init__(self, out_unit=2):
        super(MyCNN, self).__init__(
            # Convolution2D(in_channels,out_channels,ksize,stride,pad)
            conv1_1=L.Convolution2D(None, 32, 3, 1, 1),
            conv1_2=L.Convolution2D(32, 32, 3, 1, 1),
            conv2=L.Convolution2D(32, 32, 3, 1, 1),
            fc3=L.Linear(None, 256),
            fc4=L.Linear(256, out_unit)

        )

    def __call__(self, x):
        h1_1 = F.relu(self.conv1_1(x))
        h1_2 = F.max_pooling_2d(F.relu(self.conv1_2(h1_1)), 2, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1_2)), 2, 2)
        h3 = F.dropout(F.relu(self.fc3(h2)), ratio=0.5)

        return self.fc4(h3)