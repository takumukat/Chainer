# -*- coding: utf-8 -*-

import os
import chainer
import numpy as np
import matplotlib.pyplot as plt
import chainer.links as L
import chainer.functions as F
from chainer.datasets import tuple_dataset
from chainer import iterators
from chainer import serializers
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer import cuda
from cnn_original import MyCNN

#
def makeIterator(train_path, test_path, batchsize):
    train_dataset = np.load(train_path)
    test_dataset = np.load(test_path)
    train = tuple_dataset.TupleDataset(train_dataset.transpose(1,0)[0], train_dataset.transpose(1,0)[1])
    test = tuple_dataset.TupleDataset(test_dataset.transpose(1,0)[0], test_dataset.transpose(1,0)[1])
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    print('train: {}'.format(len(train_dataset)))
    print('test:  {}'.format(len(test_dataset)))

    return train_iter, test_iter


# モデル
class CNN(chainer.Chain):
    #------------------------------------------rightleft--result--------------------------------------
    #epoch: 50  train_loss: 0.0000  train_accuracy: 1.0000   test_loss: 0.0808   test_accuracy: 0.9900  <--1300
    #epoch: 50  train_loss: 0.0004  train_accuracy: 1.0000   test_loss: 0.0680   test_accuracy: 0.9900  <--1300   drop out
    #epoch: 50  train_loss: 0.0000  train_accuracy: 1.0000   test_loss: 0.0166   test_accuracy: 0.9967  <--1300-2
    #epoch: 50  train_loss: 0.0002  train_accuracy: 1.0000   test_loss: 0.0044   test_accuracy: 0.9967  <--1300-2 drop out
    #epoch: 50  train_loss: 0.0001  train_accuracy: 1.0000   test_loss: 0.0173   test_accuracy: 0.9909  <--1700   drop out
    #epoch: 50  train_loss: 0.0000  train_accuracy: 1.0000   test_loss: 0.0148   test_accuracy: 0.9956  <--3000   drop out
    #-------------------------------------------------------------------------------------------------

    def __init__(self, out_unit=2):
        super(CNN, self).__init__(
            # Convolution2D(in_channels,out_channels,ksize,stride,pad)
            conv1=L.Convolution2D(None, 32, 5, 1, 0),
            conv2=L.Convolution2D(32, 32, 5, 1, 0),
            conv3=L.Convolution2D(32, 64, 5, 1, 0),
            fc4=L.Linear(None, 64),
            fc5=L.Linear(64, out_unit)

        )

    def __call__(self, x):
        h1 = self.conv1(x)
        h2 = F.max_pooling_2d(h1, ksize=3, stride=2)
        h3 = F.relu(h2)
        h4 = self.conv2(h3)
        h5 = F.relu(h4)
        h6 = F.average_pooling_2d(h5, ksize=3, stride=2)
        h7 = self.conv3(h6)
        h8 = F.relu(h7)
        h9 = F.average_pooling_2d(h8, ksize=3, stride=2)
        #h10 = F.dropout(self.fc4(h9), ratio=0.5)
        h10 = self.fc4(h9)

        return self.fc5(h10)


def training(model_object, gpu_id, max_epoch, train_iterator, test_iterator):
    model = model_object
    train_iter = train_iterator
    test_iter = test_iterator

    if gpu_id >= 0:
        try:
            cuda.get_device(gpu_id).use()
            model.to_gpu(gpu_id)
        except:
            print(os.environ["PATH"])
            os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
            cuda.get_device(gpu_id).use()
            model.to_gpu(gpu_id)

    optimizer = optimizers.Adam()
    optimizer.setup(model)


    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []

    while train_iter.epoch < max_epoch:
        # --------------学習の1イテレーション-----------------
        train_batch = train_iter.next()
        x, t = concat_examples(train_batch, gpu_id)

        y = model(x)

        # loss, accuracyの計算
        train_loss = F.softmax_cross_entropy(y, t)
        train_accuracy = F.accuracy(y, t)

        # 勾配の計算
        model.cleargrads()
        train_loss.backward()

        # パラメータの更新
        optimizer.update()

        # --------------------ここまで------------------------

        # 1エポックごとに予測精度を測り、汎化性能の向上をチェック
        if train_iter.is_new_epoch:  # 1エポックが終わったら

            # ロスの表示
            trainLoss.append(cuda.to_cpu(train_loss.data))
            trainAccuracy.append(cuda.to_cpu(train_accuracy.data))
            print('epoch: {:02d}  train_loss: {:.04f}  train_accuracy: {:.04f}   '.format(
                train_iter.epoch, float(cuda.to_cpu(train_loss.data)), float(cuda.to_cpu(train_accuracy.data))), end='')

            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                x_test, t_test = concat_examples(test_batch, gpu_id)

                # テストデータをforward
                y_test = model(x_test)

                # ロスを計算
                loss_test = F.softmax_cross_entropy(y_test, t_test)
                test_losses.append(cuda.to_cpu(loss_test.data))

                # 精度を計算
                test_accuracy = F.accuracy(y_test, t_test)
                test_accuracy.to_cpu()
                test_accuracies.append(test_accuracy.data)

                #test: 1 epoch only
                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            testLoss.append(np.mean(test_losses))
            testAccuracy.append(np.mean(test_accuracies))
            print('test_loss: {:.04f}   test_accuracy: {:.04f}'.format(
                np.mean(test_losses), np.mean(test_accuracies)))  # テスト結果のロスと精度の平均を表示

    model.to_cpu()
    return model, trainLoss, trainAccuracy, testLoss, testAccuracy



#------------------save graph--------------------

def save_graph(max_epoch, trainLoss, trainAccuracy, testLoss, testAccuracy, filename_loss, filename_accuracy):
    fnameLoss = filename_loss
    fnameAccuracy = filename_accuracy
    x = np.arange(1,max_epoch+1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(x, trainLoss)
    ax1.plot(x, testLoss)
    plt.legend(['train loss', 'test loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(fnameLoss)
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(x, trainAccuracy)
    ax2.plot(x, testAccuracy)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(fnameAccuracy)
    plt.show()


def main():
    dataName = 'rightleft3000'
    train_path = 'datasets/tupple_datasets/'+dataName+'_train.npy'
    test_path = 'datasets/tupple_datasets/'+dataName+'_test.npy'
    max_epoch = 50
    batchsize = 50
    gpu_id = -1
    fname_loss = 'rightleft_result/'+dataName+'_{0:03d}_2_loss.png'.format(max_epoch)
    fname_accuracy = 'rightleft_result/'+dataName+'_{0:03d}_2_accuracy.png'.format(max_epoch)

    #original_model = MyCNN()   #vgg16like CNN


    train_iter, test_iter = makeIterator(train_path, test_path ,batchsize)

    model, trainLoss, trainAccuracy, testLoss, testAccuracy = training(model_object=CNN(), gpu_id=gpu_id, max_epoch=max_epoch, train_iterator=train_iter, test_iterator=test_iter)

    serializers.save_npz('model/'+dataName+'_{0:03d}_2.model'.format(max_epoch), model)

    save_graph(max_epoch=max_epoch, trainLoss=trainLoss, trainAccuracy=trainAccuracy, testLoss= testLoss, testAccuracy=testAccuracy, filename_loss=fname_loss, filename_accuracy=fname_accuracy)



if __name__ == '__main__':

    main()
