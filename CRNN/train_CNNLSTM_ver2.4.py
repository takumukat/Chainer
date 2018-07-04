# -*- coding: utf-8 -*-
# behavior recognition by CRNN
# 動画の最後のフレームで正解，不正解を判断する

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import optimizers
from chainer import Variable
from chainer import cuda
from CNNLSTMmodel_original import CNNLSTM as original
from CRNN_model_ver2 import CRNN

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2
import csv
from make_dataset import MakeData

#training loop
def trainLoop(maxEpoch, batchSize, modelObj, train, test, videoDict, gpuID, folderName):

    model = modelObj
    trainData = np.array(train)
    testData = np.array(test)

    print('len(train video): {}'.format(len(trainData)))
    print('len( test video): {}'.format(len(testData)))

    if gpuID >= 0:
        try:
            xp = cuda.cupy
            cuda.get_device(gpuID).use()
            model.to_gpu(gpuID)
            print('using GPU')
        except:
            xp = np
            print('not using GPU')
    else:
        xp = np
        print('not using GPU')

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # 勾配の発散を抑える
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))


    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []
    iterTrainLoss = []
    iterTrainAccuracy = []
    iterTestLoss = []
    iterTestAccuracy = []


    # log
    f = open(folderName + 'log.txt', 'a')
    trainCSV = open(folderName + 'train.csv', 'w')
    testCSV = open(folderName + 'test.csv', 'w')
    trainwriter = csv.writer(trainCSV)
    testwriter = csv.writer(testCSV)

    # 時間測る
    start = time.time()

    # 1 [epoch] ~ max_epoch [epoch]
    for epoch in range(1, maxEpoch + 1):
        #エポックごとにデータを混ぜる
        permTrainData = np.random.permutation(trainData)
        permTestData = np.random.permutation(testData)

        iterLoss = []
        iterAcc = []
        _iterLoss = []
        _iterAcc = []

        # train
        for N in range(0, len(permTrainData), batchSize):
                # make mini batch
                if N+batchSize > len(permTrainData):
                    pass
                else:
                    trainVideoBatch, trainLossLabelBatch, trainAccLabelBatch = permTrainData[N:N+batchSize, 0], permTrainData[N:N+batchSize, 1], permTrainData[N:N+batchSize, 2]

                # 勾配を初期化
                model.cleargrads()

                # ミニバッチ
                xBatch = [Variable(xp.array(seq, dtype=xp.float32)) for seq in trainVideoBatch]
                tLossBatch = [Variable(xp.array(l_label, dtype=xp.int32)) for l_label in trainLossLabelBatch]
                tAccBatch = [Variable(xp.array(a_label, dtype=xp.int32)) for a_label in trainAccLabelBatch]

                yBatch = model(None, None, xBatch)

                loss = None
                acc = []
                for y, tLoss, tAcc in zip(yBatch, tLossBatch, tAccBatch):

                    if loss is not None:
                        loss += F.softmax_cross_entropy(y, tLoss)
                    else:
                        loss = F.softmax_cross_entropy(y, tLoss)

                    acc.append(cuda.to_cpu(F.accuracy(y, tAcc, ignore_label=-1).data))

                iterLoss.append(cuda.to_cpu(loss.data) / batchSize)
                iterAcc.append(np.mean(acc))

                iterTrainLoss.append(cuda.to_cpu(loss.data) / batchSize)
                iterTrainAccuracy.append(np.mean(acc))

                # ロスを逆伝播
                loss.backward()
                #loss.unchain_backward()

                # 重みを更新
                optimizer.update()

        # test
        for N in range(0, len(permTestData), batchSize):

            if N+batchSize > len(permTestData):
                pass
            else:
                testVideoBatch, testLossLabelBatch, testAccLabelBatch = permTestData[N:N+batchSize,0], permTestData[N:N+batchSize,1], permTestData[N:N+batchSize,2]

            _xBatch = [Variable(xp.array(seq, dtype=xp.float32)) for seq in testVideoBatch]
            _tLossBatch = [Variable(xp.array(_l_label, dtype=xp.int32)) for _l_label in testLossLabelBatch]
            _tAccBatch = [Variable(xp.array(_a_label, dtype=xp.int32)) for _a_label in testAccLabelBatch]

            _yBatch = model(None, None, _xBatch)

            _loss = None
            _acc = []

            for (i, _y), _tLoss, _tAcc in zip(enumerate(_yBatch), _tLossBatch, _tAccBatch):
                if _loss is not None:
                    _loss += F.softmax_cross_entropy(_y, _tLoss)
                else:
                    _loss = F.softmax_cross_entropy(_y, _tLoss)

                _acc.append(cuda.to_cpu(F.accuracy(_y, _tAcc, ignore_label=-1).data))

                if epoch > 40:
                    # 最後の1フレームが間違ってたら動画の名前を表示
                    if np.argmax( np.array(cuda.to_cpu(_y[-1].data)) ) != _tAcc[-1].data:
                        key = np.array(cuda.to_cpu(_xBatch[i].data), dtype=np.float32)
                        try:
                            f.write('{0:03d}  {1}\n'.format(epoch, videoDict[key.tobytes()].split('/')[-3:-1]))
                            print(videoDict[key.tobytes()].split('/')[-3:-1])
                        except:
                            pass

            _iterLoss.append(cuda.to_cpu(_loss.data) / batchSize)
            _iterAcc.append(np.mean(_acc))

            iterTestLoss.append(cuda.to_cpu(_loss.data) / batchSize)
            iterTestAccuracy.append(np.mean(_acc))

        #1動画の平均誤差,平均正解率
        epTrainLoss = np.mean(iterLoss)
        epTestLoss = np.mean(_iterLoss)

        epTrainAccuracy = np.mean(iterAcc)
        epTestAccuracy = np.mean(_iterAcc)

        #テキスト書き込み，グラフ作成用にエポックごとの誤差，正解率を格納する

        trainLoss.append(epTrainLoss)
        trainAccuracy.append(epTrainAccuracy)
        testLoss.append(epTestLoss)
        testAccuracy.append(epTestAccuracy)

        print("epoch: {:03d}    ".format(epoch), end="")
        print("trainloss: {:.04f}   ".format(epTrainLoss), end="")
        print("trainaccuracy: {:.04f}   ".format(epTrainAccuracy), end="")
        print("testloss: {:.04f}   ".format(epTestLoss), end="")
        print("testaccuracy: {:.04f}".format(epTestAccuracy))

        f.write('epoch: {0:03d}  train/loss: {1:.05f}  train/accuracy: {2:.05f}  test/loss: {3:.05f}  test/accuracy: {4:.05f}\n'.format(
            epoch, epTrainLoss, epTrainAccuracy, epTestLoss, epTestAccuracy) )

        #early stopping
        if epTrainLoss < 0.1 and epTestAccuracy < 0.95:
            break

    # かかった時間 = 終了時間 - 開始時間
    elapsed_time = time.time() - start
    print('time: {0:03d}min{1:02d}sec'.format(int(elapsed_time / 60), int(elapsed_time % 60)))
    f.write('time: {0:03d}min{1:02d}sec'.format(int(elapsed_time / 60), int(elapsed_time % 60)))
    f.close()

    trainwriter.writerow(iterTrainLoss)
    trainwriter.writerow(iterTrainAccuracy)
    testwriter.writerow(iterTestLoss)
    testwriter.writerow(iterTestAccuracy)
    trainCSV.close()
    testCSV.close()


    #重みをセーブ
    model.to_cpu()
    serializers.save_npz('model/CNNLSTM_ep{0:03d}_bs{1:03d}_fallDetection_original_2m4m6m_{2:.04f}.model'.format(epoch, batchSize, epTestAccuracy), model)

    return trainLoss, trainAccuracy, testLoss, testAccuracy, iterTrainLoss, iterTrainAccuracy



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
    plt.savefig(fnameLoss+'.png')
    #plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(x, trainAccuracy)
    ax2.plot(x, testAccuracy)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(fnameAccuracy+'.png')
    #plt.show()

def save_graph_iter(iterloss, iteracc, epoch, iterlossdata, epochlossdata, iteraccdata, epochaccdata, filename_loss, filename_acc):
    fnameLoss = filename_loss
    fnameAcc = filename_acc

    x1 = np.arange(1,iterloss+1)
    x2 = np.arange(1, iterloss, int(iterloss/epoch))
    _x1 = np.arange(1, iteracc + 1)
    _x2 = np.arange(1, iteracc, int(iteracc / epoch))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(x1, iterlossdata)
    ax1.plot(x2, epochlossdata)
    plt.legend(['train loss', 'test loss'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(fnameLoss+'_iter.png')
    #plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(_x1, iteraccdata)
    ax2.plot(_x2, epochaccdata)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.savefig(fnameAcc+'_iter.png')
    #plt.show()


def main():
    # -------------------------------------setting----------------------------------------------------
    #PATH = '/home/hashimoto/PycharmProjects/AI/datasets/FALL_DETECTION/'
    #aPATH = '/home/hashimoto/PycharmProjects/AI/datasets/dataset_2m6m/train/'
    aPATH = '/home/hashimoto/PycharmProjects/AI/datasets/dataset_2m3m4m5m6m/train2m3m4m5m6m/'
    bPATH = '/home/hashimoto/PycharmProjects/AI/datasets/dataset_2m3m5m6m/test/'
    max_epoch = 40
    batch_size = 50
    result = '/home/hashimoto/PycharmProjects/AI/CNNLSTM_result/'
    gpuID = 1
    # -------------------------------------------------------------------------------------------------

    print('epoch: {}'.format(max_epoch))
    print('batch size: {}'.format(batch_size))

    dayTime = datetime.datetime.today()
    foldername = result + dayTime.strftime("%Y_%m_%d_%H:%M:%S") + "_ver2.4_original_fallDetection_2m4m6m/"
    os.mkdir(foldername)

    fnameLoss = foldername + '_loss_fall_ep{0:03d}_bs{1:02d}'.format(max_epoch, batch_size)
    fnameAccuracy = foldername + '_accuracy_fall_ep{0:03d}_bs{1:02d}'.format(max_epoch, batch_size)

    makeDataA = MakeData(aPATH)
    makeDataB = MakeData(bPATH)
    #train, test = makeData.dataset([0,1,1], [0,0,2,0,0], 0.8)
    train, _a = makeDataA.dataset([0, 1, 1], [1,1,1,1,1], 1.0)
    _b, test = makeDataB.dataset([0,1,1], [2,2,2,2,2], 0.0)


    modelObj = original(out_unit=2)
    #modelObj = CRNN(out_unit=2)


    videoDict = makeDataB.makeDict()

    trainLoss, trainAccuracy, testLoss, testAccuracy, iterTrainLoss, iterTrainAccuracy = trainLoop(max_epoch, batch_size, modelObj, train, test, videoDict, gpuID=gpuID, folderName=foldername)
    save_graph(len(testAccuracy), trainLoss, trainAccuracy, testLoss, testAccuracy, fnameLoss, fnameAccuracy)
    save_graph_iter(len(iterTrainLoss), len(iterTrainAccuracy), len(testAccuracy), iterTrainLoss, testLoss, iterTrainAccuracy, testAccuracy, fnameLoss, fnameAccuracy)

    return testAccuracy


if __name__ == '__main__':
    #main()
    #"""
    lim = 5
    count = 0
    while count <= lim:
        testacc = main()
        print('last test accuracy: {}'.format(testacc[-1]))
        if len(testacc) < 0:
            count += 1
    """
    while True:
        testAcc =main()
        print(testAcc[-1])

        if testAcc[-1] > 0.8:
            break
    """