# -*- coding: utf-8 -*-
#sin(x), sin(2x), sin(3x)の分類をLSTMでやる

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import optimizers
from chainer.cuda import to_cpu
from chainer import Variable

import math
import random
import time
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import csv


class RNN(chainer.Chain):
    def __init__(self, layer, lstm_units, fc_units=3, out_units=3):
        super(RNN, self).__init__(
            fc1 = L.Linear(None, lstm_units),
            lstm2 = L.LSTM(lstm_units,lstm_units),
            lstm3 = L.LSTM(lstm_units,lstm_units),
            lstm4 = L.LSTM(lstm_units,lstm_units),
            lstm_last = L.LSTM(lstm_units, fc_units),
            fc4 = L.Linear(fc_units, out_units)
        )
        layer = layer

    def reset_state(self):
        if layer == 1:
            self.lstm_last.reset_state()

        elif layer == 2:
            self.lstm2.reset_state()
            self.lstm_last.reset_state()

        elif layer == 3:
            self.lstm2.reset_state()
            self.lstm3.reset_state()
            self.lstm_last.reset_state()

        elif layer == 4:
            self.lstm2.reset_state()
            self.lstm3.reset_state()
            self.lstm4.reset_state()
            self.lstm_last.reset_state()

        self.cleargrads()

    def forward(self, layer, x):
        h1 = self.fc1(x)
        if layer == 1:
            h5 = self.lstm_last(h1)

        elif layer == 2:
            h4 = self.lstm2(h1)
            h5 = self.lstm_last(h4)

        elif layer == 3:
            h3 = self.lstm2(h1)
            h4 = self.lstm3(h3)
            h5 = self.lstm_last(h4)

        elif layer == 4:
            h2 = self.lstm2(h1)
            h3 = self.lstm3(h2)
            h4 = self.lstm4(h3)
            h5 = self.lstm_last(h4)

        h6 = self.fc4(h5)

        return h6

    def __call__(self, x):
        y = self.forward(layer, x)

        return y


class DataMaker(object):
    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles

    def make(self, n, noise):
        #sin(n*2π)をnumber_of_cycles個連結させる
        sin_nx = np.array([math.sin(n * i * 2 * math.pi / self.steps_per_cycle) for i in
                           range(self.steps_per_cycle)] * self.number_of_cycles)
        if noise:
            # ノイズ: rand(0～1)×2-1で-1～1を生成し、その0.1倍した数を足す
            n = (np.random.rand(self.steps_per_cycle * self.number_of_cycles) * 2 - 1) * 0.1
            return sin_nx + n
        else:
            return sin_nx


    def predictdata(self, noise):
        sinx = np.array([math.sin(i * 2 * math.pi / self.steps_per_cycle) for i in range(self.steps_per_cycle)])
        sin2x = np.array([math.sin(2 * i * 2 * math.pi / self.steps_per_cycle) for i in range(self.steps_per_cycle)])
        sin3x = np.array([math.sin(3 * i * 2 * math.pi / self.steps_per_cycle) for i in range(self.steps_per_cycle)])
        label = np.array([0,1,2])

        if noise:
            #ノイズ: rand(0～1)×2-1で-1～1を生成し、その0.1倍した数を足す
            n = (np.random.rand(self.steps_per_cycle) *2 -1) * 0.1
            return np.array([sinx+n, sin2x+n, sin3x+n]), label
        else:
            return np.array([sinx, sin2x, sin3x]), label


    def make_mini_batch(self, sinwave, mini_batch_size, length_of_sequence):
        #バッチサイズ×シーケンス長の配列を作る
        sequences = np.zeros([mini_batch_size, length_of_sequence], dtype=np.float32)
        #ラベルは1次元
        label = np.zeros([mini_batch_size], dtype=np.int32)

        for i in range(mini_batch_size):
            #sinの周波数を3つの中からランダムに選ぶ
            frequency = random.randint(0, 2)
            label[i] = frequency
            #シーケンス長の分あけてバッチの開始位置をランダムに選ぶ
            index = random.randint(0, len(sinwave[frequency]) - length_of_sequence)
            #"開始位置"～"開始位置＋シーケンス長"のsinの値を代入
            sequences[i] = sinwave[frequency][index:index + length_of_sequence]
        return sequences, label



def training(MAX_EPOCH, MINI_BATCH_SIZE, STEPS_PER_CYCLE, LENGTH_OF_SEQUENCE, NUMBER_OF_CYCLES, ALL_DATA_SIZE, data_maker, long_sin, long_sin_n, folderName, node, layer):
    # ---------------------モデル定義----------------------
    model = RNN(layer=layer, lstm_units=node)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # 勾配の発散を抑える
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
    # -----------------------------------------------------

    #sinx,sin2x,sin3xを連結させる(途中経過を見るとき用)
    #sin_x2x3x = np.concatenate((test_sequences[0], test_sequences[1], test_sequences[2]), axis=0)

    #エポックを1から数える
    epoch = 1

    #グラフ用配列
    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []

    #log
    f = open(folderName+'log.txt', 'a')

    # 時間測る
    start = time.time()
#---------------------------------------------------------------------epoch---------------------------------------------------------------------------
    while epoch <= MAX_EPOCH:
        #確率的に全データをなめたら1エポック
        for batch in range(0, ALL_DATA_SIZE, MINI_BATCH_SIZE):
            train_loss = 0
            train_accuracy = 0

            #新しいバッチを作成
            train_sequences, train_label = data_maker.make_mini_batch(long_sin, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)

            #LSTMメモリ，勾配をクリア
            model.reset_state()

            #ミニバッチ学習
            for i in range(LENGTH_OF_SEQUENCE):
                #バッチのi列目を取り出す
                x = Variable( np.asarray([ train_sequences[j, i] for j in range(MINI_BATCH_SIZE) ], dtype=np.float32) )
                t = Variable( np.asarray(train_label, dtype=np.int32))

                y = model(x.reshape(MINI_BATCH_SIZE, 1))

                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)

                #シーケンス長分足してから逆伝播
                train_loss += loss
                train_accuracy += accuracy.data

            #ロス逆伝播、勾配更新
            train_loss.backward()
            optimizer.update()

        #result  loss, accuracyをシーケンス長で割って1シーケンス分の値を出す
        print('epoch: {0:02d}  train/loss: {1:.04f}  train/accuracy: {2:.04f}'.format(epoch, float(train_loss.data / LENGTH_OF_SEQUENCE),float(train_accuracy / LENGTH_OF_SEQUENCE)), end='' )
        f.write('epoch: {0:02d}  train/loss: {1:.04f}  train/accuracy: {2:.04f}'.format(epoch, float(train_loss.data / LENGTH_OF_SEQUENCE),float(train_accuracy / LENGTH_OF_SEQUENCE)))
        #グラフ用に保存
        trainLoss.append(train_loss.data / LENGTH_OF_SEQUENCE)
        trainAccuracy.append(train_accuracy / LENGTH_OF_SEQUENCE)

        #---------------1エポックごとにテスト--------------
        #テストは1バッチだけ
        test_loss = 0
        test_accuracy = 0
        # AIの入力、回答を入れる(途中経過用)
        input_sin_wave = []
        correctLabel = []
        predict0Label = []
        predict1Label = []
        predict2Label = []

        #ノイズ入りsinでバッチを作成
        test_sequences, test_label = data_maker.make_mini_batch(long_sin_n, mini_batch_size=MINI_BATCH_SIZE,
                                                                length_of_sequence=LENGTH_OF_SEQUENCE)
        #テストバッチの0,1,2番目,正解ラベルを保存(途中経過用)
        input_sin_wave.extend(test_sequences[0])
        input_sin_wave.extend(test_sequences[1])
        input_sin_wave.extend(test_sequences[2])
        correctLabel.extend(np.array([test_label[0]] * LENGTH_OF_SEQUENCE))
        correctLabel.extend(np.array([test_label[1]] * LENGTH_OF_SEQUENCE))
        correctLabel.extend(np.array([test_label[2]] * LENGTH_OF_SEQUENCE))

        #LSTMメモリ，勾配をクリア
        model.reset_state()

        #ミニバッチテスト
        for i in range(LENGTH_OF_SEQUENCE):
            #バッチのi列目を取り出す
            x = Variable( np.asarray([ test_sequences[j, i] for j in range(MINI_BATCH_SIZE) ], dtype=np.float32) )
            t = Variable( np.asarray(test_label, dtype=np.int32))

            y = model(x.reshape(MINI_BATCH_SIZE, 1))
            # 回答の0,1,2番目を保存(途中経過用)
            predict0Label.append(y.data[0].argmax())
            predict1Label.append(y.data[1].argmax())
            predict2Label.append(y.data[2].argmax())

            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)

            #シーケンス長分足す
            test_loss += loss.data
            test_accuracy += accuracy.data

        # result  loss, accuracyをシーケンス長で割って1シーケンス分の値を出す
        print('  test/loss: {0:.04f}  test/accuracy: {1:.04f}'.format(float(test_loss / LENGTH_OF_SEQUENCE),
                                                                              float(test_accuracy / LENGTH_OF_SEQUENCE)))
        f.write('  test/loss: {0:.04f}  test/accuracy: {1:.04f}\n'.format(float(test_loss / LENGTH_OF_SEQUENCE),
                                                                              float(test_accuracy / LENGTH_OF_SEQUENCE)))
        # グラフ用
        testLoss.append(test_loss / LENGTH_OF_SEQUENCE)
        testAccuracy.append(test_accuracy / LENGTH_OF_SEQUENCE)


        """
        #test_sequences[0]はsinx, sin2x, sin3xの3つだから 3
        for n in range(test_sequences.shape[0]):
            #LSTMメモリをリセットせなうまくいかん
            model.reset_state()

            #バッチじゃなくて1個ずつ入れていく
            for i in range(LENGTH_OF_SEQUENCE):
                x = Variable( np.asarray([test_sequences[n, i]], dtype=np.float32) )
                t = Variable( np.asarray([test_label[n]], dtype=np.int32) )

                y = model(x.reshape(1,1))
                #回答を保存
                predictLabel.append(y.data.argmax())

                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)

                test_loss += loss.data
                test_accuracy += accuracy.data
        """
        #20エポックごとに途中経過グラフを保存
        if epoch % 20 == 0:
            predictLabel = []
            predictLabel.extend(predict0Label)
            predictLabel.extend(predict1Label)
            predictLabel.extend(predict2Label)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.plot(input_sin_wave)
            ax1.plot(predictLabel)
            ax1.plot(correctLabel, linestyle="--")
            #わかりやすいように線引く
            ax1.plot([50, 50], [-1, 2], linestyle="--", color='black')
            ax1.plot([100, 100], [-1, 2], linestyle="--", color='black')

            ax1.set_xlabel('t')
            ax1.set_ylabel('sin(t)')

            ax1.legend(['input_sin_wave', 'output_label', 'correct_label'])
            plt.savefig(folderName + 'sinLSTM_n{0}_layer{1}_epoch{2}.png'.format(
                node, layer, epoch))

        epoch += 1
#-------------------------------------------------------------------fin epoch-------------------------------------------------------------------------

    #かかった時間 = 終了時間 - 開始時間
    elapsed_time = time.time() - start
    print( 'time: {0:03d}分{1:02d}秒'.format(int(elapsed_time/60), int(elapsed_time%60)) )
    f.write('time: {0:03d}分{1:02d}秒'.format(int(elapsed_time/60), int(elapsed_time%60)) )
    f.close()

    return trainLoss, trainAccuracy, testLoss, testAccuracy



def save_graph(max_epoch, trainLoss, trainAccuracy, testLoss, testAccuracy, filename_loss, filename_accuracy):
    fnameLoss = filename_loss
    fnameAccuracy = filename_accuracy
    x = np.arange(1,max_epoch+1)

    #Lossのグラフ
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(x, trainLoss)
    ax1.plot(x, testLoss)
    plt.legend(['train loss', 'test loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(fnameLoss)
    #plt.show()

    #Accuracyのグラフ
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(x, trainAccuracy)
    ax2.plot(x, testAccuracy)
    plt.legend(['train accuracy', 'test accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(fnameAccuracy)
    #plt.show()


def main(node, layer):
    # エポック数
    MAX_EPOCH = 200
    # ミニバッチサイズ
    MINI_BATCH_SIZE = 20
    # 1周期の点の数
    STEPS_PER_CYCLE = 50
    # 学習の際に使うsin波の長さ(1周期分）
    LENGTH_OF_SEQUENCE = STEPS_PER_CYCLE
    # 連結するsin波の数
    NUMBER_OF_CYCLES = 10
    # 全データ数(50点×10波×3種類 = )
    ALL_DATA_SIZE = STEPS_PER_CYCLE * NUMBER_OF_CYCLES * 3

    #保存フォルダ名、グラフのファイル名
    folderName = 'sinLSTM_result/sinLSTM_n{0}_layer{1}/'.format(node, layer)
    lossImgName = folderName+'sinLSTM_loss_n{0}_layer{1}.png'.format(node, layer)
    accuracyImgName = folderName+'sinLSTM_accuracy_n{0}_layer{1}.png'.format(node, layer)

    #保存用フォルダ
    os.mkdir(folderName)

    #-----------------------long_sin(訓練バッチ作成用の長いsin波)をつくる---------------------
    random.seed(0)
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    # 各周波数の長い(NUMBER_OF_CYCLES分連結させた)sin波を作成
    # 訓練用ノイズなしsin
    sin_x = data_maker.make(1, noise=False)
    sin_2x = data_maker.make(2, noise=False)
    sin_3x = data_maker.make(3, noise=False)
    long_sin = np.array([sin_x, sin_2x, sin_3x])

    # テスト用ノイズありsin
    #sin_x_n = data_maker.make(1, noise=True)
    #sin_2x_n = data_maker.make(2, noise=True)
    #sin_3x_n = data_maker.make(3, noise=True)
    #long_sin_n = np.array([sin_x_n, sin_2x_n, sin_3x_n])
    # 検証用ノイズありsin
    #predict_sequences, predict_label = data_maker.predictdata(noise=True)
    #-----------------------------------------------------------------------------------------

    #設定を表示
    print('max epoch: {}'.format(MAX_EPOCH))
    print('batch size: {}'.format(MINI_BATCH_SIZE))
    print('1サイクル {}点'.format(STEPS_PER_CYCLE))
    print('連結するsin波の数: {}波'.format(NUMBER_OF_CYCLES))
    print('全データ数: {}点×{}波×3 = {}'.format(STEPS_PER_CYCLE, NUMBER_OF_CYCLES, ALL_DATA_SIZE))
    print('ノード：{}レイヤ：{}'.format(node, layer))

    #訓練
    trainLoss, trainAccuracy, testLoss, testAccuracy = training(
        MAX_EPOCH, MINI_BATCH_SIZE, STEPS_PER_CYCLE, LENGTH_OF_SEQUENCE, NUMBER_OF_CYCLES, ALL_DATA_SIZE,data_maker, long_sin, long_sin, folderName, node, layer)
    #loss, accuracyのグラフ保存
    save_graph(max_epoch=MAX_EPOCH, trainLoss=trainLoss, trainAccuracy=trainAccuracy, testLoss=testLoss, testAccuracy=testAccuracy,filename_loss=lossImgName, filename_accuracy=accuracyImgName)

    return testLoss[-1], testAccuracy[-1]


if __name__ == '__main__':
    x = ['3', '10', '20', '30', '40', '50']
    y = ['1', '2', '3', '4']
    testlossAll = []
    testaccuracyAll = []
    for layer in range(1, 5):
        testloss = []
        testaccuracy = []
        loss, accuracy = main(3, layer)
        testloss.append(loss)
        testaccuracy.append(accuracy)
        for node in range(10, 51, 10):
            loss, accuracy = main(node, layer)
            testloss.append(loss)
            testaccuracy.append(accuracy)

        testlossAll.append(testloss)
        testaccuracyAll.append(testaccuracy)

    acc = np.array(testaccuracyAll, dtype=np.float16)

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.set_xlabel("node")
    ax1.set_ylabel("layer")
    ax1.set_zlabel("accuracy")

    xlabels = np.array(x)
    xpos = np.arange(xlabels.shape[0])
    ylabels = np.array(y)
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = acc
    zpos = zpos.ravel()

    dx = 0.5
    dy = 0.5
    dz = zpos

    ax1.w_xaxis.set_ticks(xpos + dx / 2.)
    ax1.w_xaxis.set_ticklabels(xlabels)

    ax1.w_yaxis.set_ticks(ypos + dy / 2.)
    ax1.w_yaxis.set_ticklabels(ylabels)

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
    plt.savefig('sinLSTM_testAccuracy_ep200_bat20.png')
    plt.show()


    with open("sinLSTM_testAccuracy_ep200_bat20.csv", "wb") as f:
        writer = csv.writer(f)

        for data_row in testaccuracyAll:
            writer.writerow(data_row)