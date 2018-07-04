# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import glob

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import cuda
from chainer import serializers
from chainer import optimizers

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
        #print(h8.data.shape)
        h9 = F.average_pooling_2d(h8, ksize=3, stride=2)
        #print(h9.data.shape)
        #h10 = F.dropout(self.fc4(h9), ratio=0.5)
        h10 = self.fc4(h9)

        return self.fc5(h10)


# PATH -> traindata, trainlabel, testdata, testlabel
def dataset(PATH):
    all = 0
    dirs = sorted(os.listdir(PATH))
    for dir in dirs:
        all += len(os.listdir(PATH+dir))
    print(all)
    # 各ディレクトリの内、何割を訓練用にするか
    train_num = int(1.0 * (all / len(dirs)))

    pathsAndLabels = []
    print(dirs)
    # ディレクトリ上から順にラベル0,1,...
    for i in range(len(dirs)):
        # pathsAndLabels : [ [ディレクトリのパス, ラベル], [ , ], [ , ], ... ]
        pathsAndLabels.append(np.asarray([PATH + dirs[i] + '/', np.int32(i)]))

    # ++++++++++++++++++++++++イメージの"パス+ラベル"の配列を作る+++++++++++++++++++++++++
    trainData = []
    testData = []
    for pathAndLabel in pathsAndLabels:  # ディレクトリごとにtrainData, testDataへ、アペンドしていく
        path = pathAndLabel[0]
        label = pathAndLabel[1]

        imagelist = glob.glob(path + "*")  # ディレクトリ内の全画像
        imagelist = np.random.permutation(imagelist)  # イメージリストをごちゃまぜにする

        for i in range(train_num):  # イメージリストの 0番目 ～ {train_num -1}番目 --> train
            trainData.append([imagelist[i], label])

        for i in range(len(imagelist) - train_num):  # train_num番目 ～ 最後 --> test
            testData.append([imagelist[train_num + i], label])

    # trainData[ [イメージのパス, ラベル], [ , ], [ , ], ... ]
    trainData = np.random.permutation(trainData)  # ラベルが[0,0,0,0,...0,1,1,1,1,...1]になってるから混ぜる
    testData = np.random.permutation(testData)

    # ----------------------------イメージの"データ+ラベル"の配列を作る---------------------------
    train_imgData = []
    train_labelData = []
    test_imgData = []
    test_labelData = []

    # 訓練用
    for pathAndLabel in trainData:
        img = np.array(Image.open(pathAndLabel[0]))
        normalizedImg = np.float32(img) / 255.0
        # imgData : ndarray([[60, 80]])
        imgData = np.asarray([normalizedImg])

        train_imgData.append(imgData)
        train_labelData.append(np.int32(pathAndLabel[1]))
    # テスト用
    for pathAndLabel in testData:
        img = np.array(Image.open(pathAndLabel[0]))
        normalizedImg = np.float32(img) / 255.0
        imgData = np.asarray([normalizedImg])

        test_imgData.append(imgData)
        test_labelData.append(np.int32(pathAndLabel[1]))

    train_imgData = np.array(train_imgData).reshape(len(train_imgData), 1, 60, 80)
    train_labelData = np.array(train_labelData)
    test_imgData = np.array(test_imgData).reshape(len(test_imgData), 1, 60, 80)
    test_labelData = np.array(test_labelData)

    # train, testの数を表示
    print("train: {}".format(len(train_imgData)))
    print("test: {}".format(len(test_imgData)))

    return train_imgData, train_labelData, test_imgData, test_labelData


def training(model_object, gpu_id, max_epoch, batchsize, Path, outputPath):
    model = model_object
    x_train, t_train, x_test, t_test = dataset(Path)
    batchsize = batchsize
    max_epoch = max_epoch
    train_length = len(x_train)
    test_length = len(x_test)

    f = open(outputPath+'log.txt', 'w')

    #----------------------------------GPU関係------------------------------------
    if gpu_id >= 0:
        xp = cuda.cupy
        try:
            chainer.cuda.get_device(0).use()
            model.to_gpu(0)
            x_train = xp.asarray(x_train)
            t_train = xp.asarray(t_train)
            x_test = xp.asarray(x_test)
            t_test = xp.asarray(t_test)
        except:
            print('Cannnot use GPU')

    #-----------------------------------------------------------------------------

    #オプティマイザの選択
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    #list for graph
    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []

    start = time.time()
    #1 [epoch] ~ max_epoch [epoch]
    for epoch in range(1, max_epoch + 1):
        #sum_~ に足しこんで最後に割る
        sum_train_loss = 0
        sum_train_accuracy = 0
        sum_test_loss = 0
        sum_test_accuracy = 0
        # rand: ミニバッチをランダムに選ぶための配列
        rand = np.random.permutation(train_length)

        #train
        for i in range(0, train_length, batchsize):

            if i+batchsize < train_length:
                x = Variable(x_train[rand[i: i+batchsize]])
                t = Variable(t_train[rand[i: i+batchsize]])
                batchSize = len(x)
            else:
                x = Variable(x_train[rand[i:]])
                t = Variable(t_train[rand[i:]])
                batchSize = len(x)

            y = model(x)

            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)

            #gradient initialize
            model.cleargrads()

            # ロスを逆伝播
            loss.backward()

            #重みを更新
            optimizer.update()

            #multiply batchsize --->  divide train_length
            sum_train_loss += cuda.to_cpu(loss.data) * batchSize
            sum_train_accuracy += cuda.to_cpu(accuracy.data) * batchSize

        #test
        """
        for i in range(0, test_length, batchsize):
            if i+batchsize < test_length:
                x = Variable(x_test[i: i+batchsize])
                t = Variable(t_test[i: i+batchsize])
                batchSize = len(x)
            else:
                x = Variable(x_test[i:])
                t = Variable(t_test[i:])
                batchSize = len(x)

            y = model(x)

            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)

            sum_test_loss += cuda.to_cpu(loss.data) * batchSize
            sum_test_accuracy += cuda.to_cpu(accuracy.data) * batchSize
        """


        #result: epoch average
        train_loss = sum_train_loss/train_length
        train_accuracy = sum_train_accuracy/train_length
        #test_loss = sum_test_loss/test_length
        #test_accuracy = sum_test_accuracy/test_length
        test_loss = 0
        test_accuracy = 0

        print(
            "epoch: {:03d}    train/loss: {:.04f}    train/accuracy: {:.04f}    test/loss: {:.04f}    test/accuracy: {:.04f}".format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy))
        f.write("epoch: {:03d}    train/loss: {:.04f}    train/accuracy: {:.04f}    test/loss: {:.04f}    test/accuracy: {:.04f}\n".format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy))

        #for graph
        trainLoss.append(train_loss)
        trainAccuracy.append(train_accuracy)
        testLoss.append(test_loss)
        testAccuracy.append(test_accuracy)

    elapsedTime = time.time() - start
    print('time: {0:03d}min{1:02d}sec'.format(int(elapsedTime / 60), int(elapsedTime % 60)))
    f.write('time: {0:03d}min{1:02d}sec'.format(int(elapsedTime / 60), int(elapsedTime % 60)))
    f.close()
    model.to_cpu()
    return model, trainLoss, trainAccuracy, testLoss, testAccuracy


#save graph [trainLoss, testLoss], [trainAccuracy, testAccuracy]
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
    """
    x_trainPath = 'datasets/dataANDlabel/{}_traindata.npy'.format(dataName)
    t_trainPath = 'datasets/dataANDlabel/{}_trainlabel.npy'.format(dataName)
    x_testPath = 'datasets/dataANDlabel/{}_testdata.npy'.format(dataName)
    t_testPath = 'datasets/dataANDlabel/{}_testlabel.npy'.format(dataName)
    """

    #PATH = 'datasets/rightleft/dataset/'
    PATH = '/home/hashimoto/PycharmProjects/AI/datasets/rightleft_2018/2000-100_3m/'
    #dataName = 'k3s3'
    dataName = 'rightleft3m2000-100'
    model_object = CNN()
    max_epoch = 100
    batchsize = 50
    gpu_id = 0
    outputPath = 'rightleft_result/{0}_ep{1}/'.format(dataName, max_epoch)

    #graph name
    fname_loss = outputPath + dataName + '_ep{0:03d}_loss.png'.format(max_epoch)
    fname_accuracy = outputPath + dataName + '_ep{0:03d}_accuracy.png'.format(max_epoch)

    try:
        os.mkdir(outputPath)
    except:
        pass

    model, trainLoss, trainAccuracy, testLoss, testAccuracy = training(model_object=model_object, gpu_id=gpu_id, max_epoch=max_epoch,
                        batchsize=batchsize, Path=PATH, outputPath=outputPath)

    serializers.save_npz('model/{0}_ep{1:03d}.model'.format(dataName, max_epoch), model)

    save_graph(max_epoch=max_epoch, trainLoss=trainLoss, trainAccuracy=trainAccuracy, testLoss=testLoss,
               testAccuracy=testAccuracy, filename_loss=fname_loss, filename_accuracy=fname_accuracy)


if __name__ == '__main__':
    main()