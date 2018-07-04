# -*- coding: utf-8 -*-
# behavior recognition by CRNN
# 動画の最後のフレームで正解，不正解を判断する

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer import cuda
from CNNLSTMmodel_original import CNNLSTM as original
from CNNLSTMmodel_original_test import CNNLSTM as testmodel
from makeData import MakeData

import datetime
import os
import numpy as np
import glob
from PIL import Image
import cv2



def validation(modelObj, testPath, videoDict, labelName, gpuID, folderName):

    #動画データとラベルの塊をtestDataに格納
    model = modelObj
    makeData = MakeData(gpuID)
    testData = makeData.dataset(testPath, train=False)
    print('len( test video): {}'.format(len(testData)))

    #正解数を記録する辞書を作成
    record = {}
    for i in range(len(labelName)):
        record[labelName[i]] = {'true':0, 'false':0}

    if gpuID >= 0:
        try:
            xp = cuda.cupy
            cuda.get_device(0).use()
            model.to_gpu(0)
            print('using GPU')
        except:
            xp = np
            print('not using GPU')
    else:
        xp = np
        print('not using GPU')

    testLoss = []
    testAccuracy = []

    # log
    f = open(folderName + 'validataion_result.txt', 'a')

    permTestData = testData if len(testData)==1 else np.random.permutation(testData)

    _iterAcc = []

    # all data in batch
    testVideoBatch, testLossLabelBatch, testAccLabelBatch = permTestData[:, 0], permTestData[:, 1], permTestData[:, 2]

    _xBatch = [Variable(xp.array(video, dtype=xp.float32)) for video in testVideoBatch]
    _tLossBatch = [Variable(xp.array(losslabel, dtype=xp.int32)) for losslabel in testLossLabelBatch]
    _tAccBatch = [Variable(xp.array(acclabel, dtype=xp.int32)) for acclabel in testAccLabelBatch]

    _yBatch = model(None, None, _xBatch)

    _loss = None

    for (i, _y), _tLoss, _tAcc in zip(enumerate(_yBatch), _tLossBatch, _tAccBatch):
        if _loss is not None:
            _loss += F.softmax_cross_entropy(_y, _tLoss)
        else:
            _loss = F.softmax_cross_entropy(_y, _tLoss)

        _iterAcc.append(cuda.to_cpu(F.accuracy(_y, _tAcc, ignore_label=-1).data))

        # 最後の1フレームが間違ってたら動画の名前を
        if np.argmax( np.array(cuda.to_cpu(_y[-1].data)) ) != cuda.to_cpu(_tAcc[-1].data):
            record[labelName[cuda.to_cpu(_tAcc[-1].data)]]['false'] += 1
            key = np.array(cuda.to_cpu(_xBatch[i].data), dtype=np.float32)
            try:
                print(videoDict[key.tobytes()].split('/')[-3:-1])
                f.write('{}     '.format(videoDict[key.tobytes()].split('/')[-3:-1]))
                f.write('{}\n'.format(labelName[np.argmax( np.array(cuda.to_cpu(_y[-1].data)) )]))
            except:
                pass

        else:
            record[labelName[cuda.to_cpu(_tAcc[-1].data)]]['true'] += 1


    _iterLoss = cuda.to_cpu(_loss.data)


    #1動画の平均誤差,平均正解率
    epTestLoss = _iterLoss / len(_yBatch)

    epTestAccuracy = np.mean(_iterAcc)

    #テキスト書き込み，グラフ作成用にエポックごとの誤差，正解率を格納する
    print("testloss: {:.04f}   ".format(epTestLoss), end="")
    print("testaccuracy: {:.04f}".format(epTestAccuracy))

    f.write('\ntest/loss: {0:.05f}  test/accuracy: {1:.05f}\n'.format(epTestLoss, epTestAccuracy) )
    for i in range(len(labelName)):
        f.write('{}    True: {},   False: {}\n'.format(labelName[i], record[labelName[i]]['true'], record[labelName[i]]['false']))


    f.close()

    #重みをセーブ
    model.to_cpu()

    return


def main():
    # -------------------------------------setting----------------------------------------------------
    PATH = 'datasets/fall_detection_270_329/'

    result = 'CNNLSTM_validation/'
    gpuID = -1
    savedModel = 'CNNLSTM_ep200_bs030_fourClasses_original.model'
    # -------------------------------------------------------------------------------------------------

    dayTime = datetime.datetime.today()
    foldername = result + dayTime.strftime("%Y_%m_%d_%H:%M:%S") + "_ver2.3_{}_{}/".format(savedModel[:-6], PATH.split('/')[-2])
    os.mkdir(foldername)

    makeData = MakeData(gpuID)
    _, test, labelName = makeData.makePath(PATH, ratio=0.0)
    #modelObj = original(out_unit=4)
    modelObj = testmodel(out_unit=4)
    serializers.load_npz('model/'+savedModel, modelObj)

    videoDict = makeData.makeDict(test)

    validation(modelObj, test, videoDict, labelName, gpuID=gpuID, folderName=foldername)


if __name__ == '__main__':
    main()

