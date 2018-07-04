# -*- coding: utf-8 -*-
#推論させて正解率を出したり結果を保存したり

import chainer
#from cnn_original import MyCNN
from chainer import serializers
from chainer import Variable
from chainer import cuda

from PIL import Image
import numpy as np
import os
from scipy.misc import toimage
import glob

#入力データを作る
# videoFlag T:動画データ，F:画像データ
class makeData(object):
    def __init__(self, videoFlag):
        self.videoFlag = videoFlag

    def makeVideoData(self, path):
        dirs = os.listdir(path)

        pathsAndLabels = []
        testPathAndLabels = []
        print('ラベル: {}'.format(dirs))
        # ディレクトリ上から順にラベル0,1,...
        for i in range(len(dirs)):
            # pathsAndLabels : [ [ディレクトリのパス, ラベル], [ , ], [ , ], ... ]
            pathsAndLabels.append(np.asarray([path + dirs[i] + '/', np.int32(i)]))


        for pathAndLabel in pathsAndLabels:
            videoPaths = os.listdir(pathAndLabel[0])
            #パスをシャッフル
            videoPaths = np.random.permutation(videoPaths)
            print('{0}: {1}    '.format(dirs[int(pathAndLabel[1])], len(videoPaths)), end="" )

            for videoPath in videoPaths:
                testPathAndLabels.append(np.asarray([pathAndLabel[0] + videoPath + '/', pathAndLabel[1]]))


        print('len( test video): {}'.format(len(testPathAndLabels)))
        testPathAndLabels = np.random.permutation(testPathAndLabels)


        img = []
        acclabel = []
        for testPathAndLabel in testPathAndLabels:
            imglist = glob.glob(testPathAndLabel[0] + '*')

            imgSeq = []
            acclabelSeq = []

            for j in range(0, len(imglist), int(len(imglist) / 15)):
                image = np.array(Image.open(imglist[j]), dtype=np.float32)
                image = image / 255.0
                imgSeq.append(image)

                acclabelSeq.append(-1)
                #
                if len(imgSeq) > 14:
                    break
            acclabelSeq[-1] = testPathAndLabel[1]
            img.append(np.asarray(imgSeq, dtype=np.float32))
            # Variable no list wo tsukuru
            acclabel.append(Variable(np.asarray(acclabelSeq, dtype=np.int32)))

        return img, acclabel


    def makePictureData(self, path):
        dirs = os.listdir(path)
        dirs = sorted(dirs)
        print(path)

        dirPathsAndLabels = []
        testPathsAndLabels = []
        print('ラベル: {}'.format(dirs))
        # ディレクトリ上から順にラベル0,1,...
        for i in range(len(dirs)):
            # dirPathsAndLabels : [ [ディレクトリのパス/, ラベル], [ , ], [ , ], ... ]
            dirPathsAndLabels.append(np.asarray([path + dirs[i] + '/', np.int32(i)]))

        for dirPathAndLabel in dirPathsAndLabels:
            imgPathList = glob.glob(dirPathAndLabel[0] + '*')
            for imgPath in imgPathList:
                testPathsAndLabels.append([imgPath, dirPathAndLabel[1]])

        testPathsAndLabels = np.random.permutation(testPathsAndLabels)


        img = []
        label = []
        for testPathAndLabel in testPathsAndLabels:
            image = np.array(Image.open(testPathAndLabel[0]))
            normalizedImg = np.float32(image) / 255.0
            imgData = np.asarray([normalizedImg])

            img.append(imgData)
            label.append(np.int32(testPathAndLabel[1]))

        test_imgData = Variable(np.asarray(img).reshape(len(img), 1, 60, 80))
        test_labelData = np.asarray(label)

        return test_imgData, test_labelData


    def __call__(self, path):
        if self.videoFlag == True:
            img, label = self.makeVideoData(path)
            return img, label

        else:
            img, label = self.makePictureData(path)
            return img, label




def infer(model, imgData, labelData, outputPath, labelName, gpuID):
    xp = cuda.cupy
    infer_model = model
    num = 0
    label = labelName
    #間違ったやつをカウントする
    miss = {0:0, 1:0}
    #正解したやつをカウントする
    correct = {0:0, 1:0}

    if gpuID >= 0:
        x = xp.asarray(imgData)
        t = xp.asarray(labelData)
        chainer.cuda.get_device(0).use()
        infer_model.to_gpu()
    else:
        x = imgData
        t = labelData

    y = infer_model(x)
    y = cuda.to_cpu(y.data)
    inferedLabel = y.argmax(axis=1)

    for i in range(len(inferedLabel)):
        if inferedLabel[i] != t[i]:
            num += 1
            miss[t[i]] += 1
            img = toimage(x[i][0].data)
            img.save(outputPath + "{0:04d}.png".format(num))

            #print("No.{}".format(i+1))
            #print('infer  : ', label[inferedLabel[i]])
            #print('correct: ', label[t[i]])
            #print('')

        else:
            correct[t[i]] += 1

    infer_model.to_cpu()
    return miss, correct


def main(dataName):
    print(dataName)
    # ラベルの名前(順番をそろえる)
    labelName = ['left', 'right']
    testDataPath = 'datasets/rightleft_2018/infer_RL_2m3m5m6m/{}/'.format(dataName)
    outputPath = 'rightleft_result/mistake2m3m5m6m2000-100_ep100/{}/'.format(dataName)
    gpuID = -1
    infer_model = CNN()
    serializers.load_npz('model/rightleft2m3m5m6m2000-100_ep100^^.model', infer_model)

    #結果を入れるディレクトリを作る
    try:
        os.makedirs(outputPath)
    except:
        pass

    #入力データつくる
    #test = np.load('datasets/tupple_datasets/rightleft700_test.npy')
    datamaker = makeData(videoFlag=False)
    img, label = datamaker(testDataPath)
    print("test image: {}".format(len(img)))
    print("test label: {}\n".format(len(label)))

    #データを推論，正解，不正解をカウント
    miss, correct = infer(infer_model, img, label, outputPath, labelName, gpuID)

    #結果を表示，accuracyとか計算
    print('---------result---------')
    print('mistake: {}/{}'.format(miss[0]+miss[1], len(label)))
    print('miss left: {}'.format(miss[0]))
    print('miss right: {}'.format(miss[1]))
    print('correct left: {}'.format(correct[0]))
    print('correct right: {}'.format(correct[1]))
    c = correct[0]+correct[1]
    all = miss[0]+miss[1]+correct[0]+correct[1]
    print('\naccuracy = {0} / {1} = {2}\n'.format(c, all, c/all))

    #結果を書いて保存
    f = open(outputPath+'infer_result.txt', 'w')
    f.write('mistake: {}/{}\n'.format(miss[0] + miss[1], len(label)))
    f.write('miss left: {}\n'.format(miss[0]))
    f.write('miss right: {}\n'.format(miss[1]))
    f.write('correct left: {}\n'.format(correct[0]))
    f.write('correct right: {}\n'.format(correct[1]))
    f.write('\naccuracy = {0} / {1} = {2}'.format(c, all, c / all))
    f.close()

if __name__ == '__main__':
    datalist = os.listdir('datasets/rightleft_2018/infer_RL_2m3m5m6m/')
    #datalist = ['rl2_5m', 'rl2m']
    print(datalist, end='\n\n')
    for data in datalist:
        main(data)