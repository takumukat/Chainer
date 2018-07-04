# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
import os
from PIL import Image
import numpy as np
from make_dataset import MakeData
from chainer import cuda
from chainer import Variable
import matplotlib.pyplot as plt


class CRNN(chainer.Chain):
    def __init__(self, out_unit):
        super(CRNN, self).__init__(
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

        return h10


    def __call__(self, hx, cx, x):
        if type(x) is not list:
            item = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            h10 = [self.CNN(item)]
            hy, cy, h11 = self.lstm5(hx, cx, h10)
            y = [self.fc6(item) for item in h11]

        else:
            h10 = [self.CNN(item.reshape(item.shape[0], 1, item.shape[1], item.shape[2])) for item in x]
            print(h10[0].shape)
            hy, cy, h11 = self.lstm5(hx, cx, h10)
            y = [self.fc6(item) for item in h11]

        return y, hy, cy



def oneFrame(videoPath):
    model = CRNN(2)
    serializers.load_npz('model/CNNLSTM_ep060_bs050_fallDetection_original.model', model)
    #serializers.load_npz('model/CNNLSTM_ep060_bs050_fallDetection_original_3mONLY.model', model)

    path = videoPath
    names = sorted(os.listdir(path))

    movie = [np.array(Image.open(path+name)) for name in names]
    movie = [np.array(movie, dtype=np.float32) / 255.0]

    y0, h, c = model(None, None, movie)

    probability0 = F.softmax(y0[0])[-1].data


    hy, cy = None, None
    for im in movie[0]:
        d = [im]
        d = np.array(d, dtype=np.float32)

        y, hy, cy = model(hy, cy, d)
        probability = F.softmax(y[0])[0].data
        print('Continuous frame  fall:%.8f%%  not fall:%.8f%%' % (probability[0], probability[1]))

    print('\n           video  fall:%.8f%%  not fall:%.8f%%\n'%(probability0[0], probability0[1]))

    return


def mixedFrame(videoPath):
    model = CRNN(2)
    serializers.load_npz('model/CNNLSTM_ep060_bs050_fallDetection_original.model', model)

    mixed = []
    path = videoPath
    dirs = sorted(os.listdir(path))
    dirs[0], dirs[1] = dirs[1], dirs[0]
    print('{} --> {} --> {}'.format(dirs[0],dirs[1],dirs[2]))
    videolen = []
    for i,dir in enumerate(dirs):
        names = sorted(os.listdir(path+dir))
        if i ==0:
            videolen.append(len(names))
        else:
            videolen.append(len(names)+videolen[i-1])
        imgs = [np.array(Image.open(path+dir+'/'+name)) for name in names]
        for img in imgs:
            mixed.append(img)

    print(videolen)
    mixed = np.array(mixed, dtype=np.float32) / 255.0

    hy, cy = None, None
    for i, img in enumerate(mixed):
        if i in videolen:
            print('---------chainge video---------')

        d = [img]
        d = np.array(d, dtype=np.float32)

        y, hy, cy = model(hy, cy, d)

        probability = F.softmax(y[0])[0].data
        print('fall:%.8f%%  not fall:%.8f%%' % (probability[0], probability[1]))

        return


def validation(modelName, testData, gpuID, folderName, videoDict):
    model = CRNN(2)
    #modelName = 'model/CNNLSTM_ep050_bs050_fallDetection_original_2m3m4m5m6m_0.9350.model'
    serializers.load_npz(modelName, model)

    test = np.array(testData)

    print('len( test video): {}'.format(len(test)))

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

    # log
    f = open(folderName + 'result.txt', 'a')
    f.write(modelName + '\n')


    testVideoBatch, testLossLabelBatch, testAccLabelBatch = test[:,0], test[:,1], test[:,2]

    _xBatch = [Variable(xp.array(seq, dtype=xp.float32)) for seq in testVideoBatch]
    _tLossBatch = [Variable(xp.array(_l_label, dtype=xp.int32)) for _l_label in testLossLabelBatch]
    _tAccBatch = [Variable(xp.array(_a_label, dtype=xp.int32)) for _a_label in testAccLabelBatch]

    _yBatch, h, c = model(None, None, _xBatch)

    _acc = []

    for (i, _y), _tLoss, _tAcc in zip(enumerate(_yBatch), _tLossBatch, _tAccBatch):
        _acc.append(cuda.to_cpu(F.accuracy(_y, _tAcc, ignore_label=-1).data))


        # 最後の1フレームが間違ってたら動画の名前を表示
        if np.argmax(np.array(cuda.to_cpu(_y[-1].data))) != _tAcc[-1].data:
            key = np.array(cuda.to_cpu(_xBatch[i].data), dtype=np.float32)
            try:
                f.write('{0}\n'.format(videoDict[key.tobytes()].split('/')[-3:-1]))
                print(videoDict[key.tobytes()].split('/')[-3:-1])
            except:
                pass

    accuracy = np.mean(_acc)
    f.write('accuracy: {}\n'.format(accuracy))
    print('accuracy: {}\n\n'.format(accuracy))

    f.close()

    return accuracy



def main(modelName):
    mix = '/home/hashimoto/PycharmProjects/AI/datasets/realtime_test/'
    mix2 = '/home/hashimoto/PycharmProjects/AI/datasets/realtime_test2/'
    mix3 = '/home/hashimoto/PycharmProjects/AI/datasets/realtime_test3/'
    fall = 'datasets/fall_detection_270_329/fall_270/180117_17_06_53_fall5m/'
    fall2 = 'datasets/fall_detection_270_329/fall_270/171120_20_40_07_fall6m_24/'
    other = 'datasets/fall_detection_270_329/other_329/1225_lie5m_0002/'
    test3m = '/home/hashimoto/PycharmProjects/AI/datasets/dataset_3mtrain/test/'
    test26 = '/home/hashimoto/PycharmProjects/AI/datasets/dataset_2m6m/test2/'
    test4 = '/home/hashimoto/PycharmProjects/AI/datasets/augdataset_4m/aug_test2/'

    gpuID = 1

    """
    #oneFrame(fall)
    #mixedFrame(mix3)
    
    dirs = os.listdir(mix)
    for dir in dirs:
        print(dir)
        oneFrame(mix+dir+'/')
    """
    result = '/home/hashimoto/PycharmProjects/AI/CNNLSTM_validation_compare/'
    testPath = test4

    dirs = sorted(os.listdir(testPath))
    accuracy = []
    os.mkdir(result+'{0:02d}'.format(len(os.listdir(result))))

    for dir in dirs:
        n = len(os.listdir(result))-1
        foldername = result + '{0:02d}/validation_fallDetection_2m4m6m_{1}/'.format(n, dir)
        os.mkdir(foldername)

        makeData = MakeData(testPath+dir+'/')

        dic = makeData.makeDict()
        _, testData = makeData.dataset([0,1,1], [2], 0)

        acc = validation(modelName, testData, gpuID, foldername, dic)
        accuracy.append(acc)

    #colorlist = ["b","b","r","b","b"]
    #colorlist = ["r", "b", "b", "b", "r"]
    #colorlist = ["b", "r", "b", "r", "b"]
    colorlist = ["r", "b", "r", "b", "r"]
    #colorlist = ["r","r","r","r","r"]

    plt.bar(range(len(dirs)), accuracy, color=colorlist, tick_label=dirs)
    plt.ylim([0, 1.0])
    plt.savefig(result+'{0:02d}/accuracy.png'.format(n))

    f = open(result+'{0:02d}/ave_min.txt'.format(n), 'w')
    average = np.mean(accuracy)
    mini = np.min(accuracy)

    f.write('average: {}\n'.format(average))
    f.write('minimun: {}\n'.format(mini))
    f.close()
    print('average: {}'.format(average))
    print('minimun: {}'.format(mini))



if __name__ == '__main__':
    #main(None)

    mlist = os.listdir('model246')
    for m in mlist:
        main('model246/'+m)