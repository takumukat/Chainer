# -*- coding: utf-8 -*-
# behavior recognition by using CNN-LSTM
# 動画の全フレームの平均値で正解，不正解を判断する

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import optimizers
from chainer import Variable
from chainer import cuda
from CNNLSTMmodel_original import CNNLSTM as original

import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


# 動画フォルダのパスをシャッフルし，行動ごとにラベル付け
def makePath(path):
    dirs = sorted(os.listdir(path))

    pathsAndLabels = []
    trainPathAndLabels = []
    testPathAndLabels = []
    print('ラベル: {}'.format(dirs))
    # ディレクトリの上からラベル0,1,...
    for i in range(len(dirs)):
        # pathsAndLabels : [ [ディレクトリのパス, ラベル], [ , ], [ , ], ... ]
        pathsAndLabels.append(np.asarray([path + dirs[i] + '/', np.int32(i)]))

    #trainPathAndLabels, testPathAndLabelsに静止画の束が入ってるフォルダへのパスを入れる
    for pathAndLabel in pathsAndLabels:
        videoPath = os.listdir(pathAndLabel[0])
        #パスをシャッフル
        videoPath = np.random.permutation(videoPath)
        print('{0}: {1}    '.format(dirs[int(pathAndLabel[1])], len(videoPath)), end="" )

        # train_numで訓練用データ数の割合を変更
        train_num = int(0.8 * len(videoPath))
        print('train: {0}   test: {1}'.format(train_num, len(videoPath)-train_num) )

        for i in range(train_num):
            trainPathAndLabels.append(np.asarray([pathAndLabel[0] + videoPath[i] + '/', pathAndLabel[1]]))

        for i in range(train_num, len(videoPath)):
            testPathAndLabels.append(np.asarray([pathAndLabel[0] + videoPath[i] + '/', pathAndLabel[1]]))

    print('len(train video): {}'.format(len(trainPathAndLabels)))
    print('len( test video): {}'.format(len(testPathAndLabels)))
    # クラス0，クラス1，...になってるからまたシャッフル
    trainPathAndLabels = np.random.permutation(trainPathAndLabels)
    testPathAndLabels = np.random.permutation(testPathAndLabels)

    return trainPathAndLabels, testPathAndLabels


#動画をキーにパスを値にした辞書を作成
def makeDict(testPath):
    videoDict = {}

    #全動画のパスから1つずつ
    for path in testPath:
        #動画の全フレームのパスを取得
        try:
            imgNameList = sorted(glob.glob(path[0] + '*'))
        except:
            continue

        imgSeq = []     #動画を入れる配列

        #フレームのデータを順々入れていく
        for imgName in imgNameList:
            image = np.array(Image.open(imgName), dtype=np.float32)
            image = image / 255.0
            imgSeq.append(image)

        #numpy arrayに変換
        imgSeq = np.array(imgSeq, dtype=np.float32)
        #byteに変換してキーとし，動画のパスを値にする
        videoDict[imgSeq.tobytes()] = path[0]

    return videoDict


#ミニバッチ作成
def makeBatch(PathAndLabels, batchsize, N, gpuID):

    batch_img = []          #データのミニバッチを入れる
    batch_losslabel = []    #全フレーム分のラベルをミニバッチ分入れる
    batch_acclabel = []     #動画の代表ラベルをミニバッチ分入れる

    #(batchsize)個の動画を１つのミニバッチにする
    for i in range(N, N+batchsize):
        #全フレームのファイル名を取得
        try:
            imgNameList = sorted(glob.glob(PathAndLabels[i][0]+'*'))
        #バッチサイズ分の動画がない場合，スルーして小さなバッチサイズになる
        except:
            continue

        #動画を格納する配列
        imgSeq = []
        #ラベルを格納する配列
        losslabelSeq = []

        #シーケンス枚数を統一せず,ミニバッチを作る
        for imgName in imgNameList:
            image = np.array(Image.open(imgName), dtype=np.float32)
            image = image / 255.0

            imgSeq.append(image)

            losslabelSeq.append(PathAndLabels[i][1])

        batch_img.append(np.array(imgSeq))

        batch_losslabel.append(losslabelSeq)
        batch_acclabel.append(PathAndLabels[i][1])

    return batch_img, batch_losslabel, batch_acclabel


#training loop
def trainLoop(maxEpoch, batchSize, modelObj, trainPath, testPath, videoDict, gpuID, folderName):

    model = modelObj

    if gpuID >= 0:
        try:
            xp = cuda.cupy
            cuda.get_device(0).use()
            model.to_gpu(0)
            print('using GPU')
        except:
            print('not using GPU')
            xp = np
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

    # ログを書き込む
    f = open(folderName + 'log.txt', 'a')

    # 時間測る
    start = time.time()

    # 1 [epoch] ~ max_epoch [epoch]
    for epoch in range(1, maxEpoch + 1):
        #エポックごとに動画のパスを混ぜる
        permTrainPath = np.random.permutation(trainPath)
        permTestPath = np.random.permutation(testPath)

        #イテレーションごとの結果が入る
        iterLoss = []
        iterAcc = []
        _iterLoss = []
        _iterAcc = []

        # train
        for N in range(0, len(permTrainPath), batchSize):
            # make mini batch
            trainVideoBatch, trainLossLabelBatch, trainAccLabelBatch = makeBatch(permTrainPath, batchSize, N, gpuID)

            # 勾配を初期化
            model.cleargrads()

            # ミニバッチ
            xBatch = [Variable(xp.array(seq, dtype=xp.float32)) for seq in trainVideoBatch]
            tLossBatch = [Variable(xp.array(l_label, dtype=xp.int32)) for l_label in trainLossLabelBatch]
            tAccBatch = xp.array(trainAccLabelBatch, dtype=xp.int32)

            yBatch = model(None, None, xBatch)

            loss = None
            acc = []
            for y, tLoss in zip(yBatch, tLossBatch):

                #全フレーム分lossを加算
                if loss is not None:
                    loss += F.softmax_cross_entropy(y, tLoss)
                else:
                    loss = F.softmax_cross_entropy(y, tLoss)

                #softmaxの値を全フレーム平均
                acc.append(np.mean(F.softmax(y).data, axis=0))

            accuracy = F.accuracy(xp.array(acc, dtype=np.float32), tAccBatch)

            iterLoss.append(cuda.to_cpu(loss.data) / batchSize)
            iterAcc.append(cuda.to_cpu(accuracy.data))

            # ロスを送E��播
            loss.backward()
            #loss.unchain_backward()

            # 重みを更新
            optimizer.update()

        # test
        for N in range(0, len(permTestPath), batchSize):
            testVideoBatch, testLossLabelBatch, testAccLabelBatch = makeBatch(permTestPath, batchSize, N, gpuID)


            _xBatch = [Variable(xp.array(seq, dtype=xp.float32)) for seq in testVideoBatch]
            _tLossBatch = [Variable(xp.array(l_label, dtype=xp.int32)) for l_label in testLossLabelBatch]
            _tAccBatch = xp.array(testAccLabelBatch, dtype=xp.int32)

            _yBatch = model(None, None, _xBatch)

            _loss = None
            _acc = []

            # i:カウンタ
            for (i, _y), _tLoss, _tAcc in zip(enumerate(_yBatch), _tLossBatch, _tAccBatch):
                if _loss is not None:
                    _loss += F.softmax_cross_entropy(_y, _tLoss)
                else:
                    _loss = F.softmax_cross_entropy(_y, _tLoss)

                _acc.append(np.mean(F.softmax(_y).data, axis=0))

            _accuracy = F.accuracy(xp.array(_acc, dtype=xp.float32), _tAccBatch)

            _iterLoss.append(cuda.to_cpu(_loss.data) / batchSize)
            _iterAcc.append(cuda.to_cpu(_accuracy.data))

            if epoch > 100:
                # 最後の1フレームが間違ってたら動画の名前を表示
                if np.argmax(cuda.to_cpu(_y[-1].data)) != _tAcc.data:
                    key = np.array(cuda.to_cpu(_xBatch[i].data), dtype=np.float32)
                    try:
                        print(videoDict[key.tobytes()].split('/')[-3:-1])
                    except:
                        pass

        #loss, accuracy のエポック平均
        epTrainLoss = np.mean(iterLoss)
        epTestLoss = np.mean(_iterLoss)

        epTrainAccuracy = np.mean(iterAcc)
        epTestAccuracy = np.mean(_iterAcc)

        #グラフ作成用にエポックごとのloss, accuracyを格納
        trainLoss.append(epTrainLoss)
        trainAccuracy.append(epTrainAccuracy)
        testLoss.append(epTestLoss)
        testAccuracy.append(epTestAccuracy)

        #エポックごとに結果を表示
        print("epoch: {:03d}    ".format(epoch), end="")
        print("trainloss: {:.04f}   ".format(epTrainLoss), end="")
        print("trainaccuracy: {:.04f}   ".format(epTrainAccuracy), end="")
        print("testloss: {:.04f}   ".format(epTestLoss), end="")
        print("testaccuracy: {:.04f}".format(epTestAccuracy))

        #テキストに結果を書き込む
        f.write('epoch: {0:03d}  train/loss: {1:.05f}  train/accuracy: {2:.05f}  test/loss: {3:.05f}  test/accuracy: {4:.05f}\n'.format(
            epoch, epTrainLoss, epTrainAccuracy, epTestLoss, epTestAccuracy) )

        #early stopping
        #if epTestLoss < 0.0001 and epTestAccuracy > 0.99:
            #break

    # かかった時間 = 終了時間 - 開始時間
    elapsed_time = time.time() - start
    print('time: {0:03d}min{1:02d}sec'.format(int(elapsed_time / 60), int(elapsed_time % 60)))
    f.write('time: {0:03d}min{1:02d}sec'.format(int(elapsed_time / 60), int(elapsed_time % 60)))
    f.close()

    #重みをセーブ
    model.to_cpu()
    serializers.save_npz('model/CNNLSTM_ep{0:03d}_bs{1:03d}_wfss0124.model'.format(epoch, batchSize), model)

    return trainLoss, trainAccuracy, testLoss, testAccuracy



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
    # -------------------------------------setting----------------------------------------------------
    PATH = '/home/hashimoto/PycharmProjects/AI/datasets/WalkFallSitStand/'
    #PATH = '/home/hashimoto/PycharmProjects/AI/datasets/fall_detection_dataset_contrast/'
    max_epoch = 200
    batch_size = 30
    result = 'CNNLSTM_result/'
    gpuID = 0
    # -------------------------------------------------------------------------------------------------

    print('epoch: {}'.format(max_epoch))
    print('batch size: {}'.format(batch_size))

    dayTime = datetime.datetime.today()
    foldername = result + dayTime.strftime("%Y_%m_%d_%H:%M:%S") + "_ver3_original_CNN/"
    os.mkdir(foldername)
    fnameLoss = foldername + 'loss_fall_ep{0:03d}_bs{1:02d}_wfss_ver3.png'.format(max_epoch, batch_size)
    fnameAccuracy = foldername + 'accuracy_fall_ep{0:03d}_bs{1:02d}_wfss_ver3.png'.format(max_epoch, batch_size)

    train, test = makePath(PATH)
    videoDict = makeDict(test)
    #modelObj = original(out_unit=4)
    modelObj = CNNalpha(out_unit=4)
    trainLoss, trainAccuracy, testLoss, testAccuracy = trainLoop(max_epoch, batch_size, modelObj, train, test, videoDict, gpuID=gpuID, folderName=foldername)
    save_graph(max_epoch, trainLoss, trainAccuracy, testLoss, testAccuracy, fnameLoss, fnameAccuracy)



if __name__ == '__main__':
    main()

