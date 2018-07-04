# coding: utf-8
#ドラゼミ用 ただ訓練するだけ


import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import Variable
from chainer.datasets import mnist
from chainer import cuda

import matplotlib.pyplot as plt
import numpy as np
import time


class MLP(chainer.Chain):
    def __init__(self, n_out=10):
        super(MLP, self).__init__(
            l1=L.Linear(None, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, n_out),
        )

    def forward(self, x):
        h1_1 = self.l1(x)
        h1_2 = F.relu(h1_1)
        h2_1 = self.l2(h1_2)
        h2_2 = F.relu(h2_1)
        y = self.l3(h2_2)

        return y


class CNN(chainer.Chain):
    def __init__(self, n_out=10):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 16, ksize=5, stride=1, pad=2),
            conv2=L.Convolution2D(16, 16, ksize=3, stride=1, pad=1),
            l3=L.Linear(None, 64),
            l4=L.Linear(64,n_out),

        )

    def forward(self, x):
        h1_1 = self.conv1(x)
        h1_2 = F.max_pooling_2d(h1_1, ksize=3, stride=3)
        h1_3 = F.relu(h1_2)
        h2_1 = self.conv2(h1_3)
        h2_2 = F.max_pooling_2d(h2_1, ksize=3, stride=3)
        h2_3 = F.relu(h2_2)
        h3_1 = self.l3(h2_3)
        h3_2 = F.relu(h3_1)
        y = self.l4(h3_2)

        return y


def trainMLP(train, test):
    train_l = []
    train_a = []
    test_l = []
    test_a = []

    #モデルのインスタンス生成
    model = MLP(10)
    #最適化手法の選択
    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)

    #ハイパーパラメータの設定
    epoch = 1
    max_epoch = 500
    batch_size = 1000
    start = time.time()

    # max epoch　まで学習
    while epoch <= max_epoch:

        # epochごとに訓練データをシャッフル
        train = np.random.permutation(train)
        loss_list = []  #誤差を入れる
        acc_list = []   #正解率を入れる

        # ミニバッチごとに重みを更新,訓練データを使い切るまでループ
        # ---------------------------------------epoch 開始---------------------------------
        for i in range(0, len(train), batch_size):
            #勾配を初期化
            model.cleargrads()

            # i番目からバッチサイズ分のデータを取ってくる
            # x:入力 t:ラベル
            x = np.vstack(train[i:i+batch_size, 0])
            t = train[i:i+batch_size, 1].astype(np.int32)

            # chainerは基本Variable型に変換してからニューラルネットに入れる
            x = Variable(x)
            t = Variable(t)

            # 出力層の値が返ってくる
            y = model.forward(x)

            # 正解率を計算
            acc = F.accuracy(y, t)
            acc_list.append(acc.data)

            # 誤差を計算
            loss = F.softmax_cross_entropy(y, t)
            loss_list.append(np.sum(loss.data))
            # 誤差から逆伝播して勾配を算出
            loss.backward()

            # 勾配からさっき選んだ最適化手法で重みを更新
            optimizer.update()

            print("{0:5d}/{1}".format(i+batch_size, len(train)))
        # --------------------------------epoch 終了---------------------------------------

        #平均誤差
        epLoss = sum(loss_list) / len(train)
        train_l.append(epLoss)
        #平均正解率
        epAcc = sum(acc_list) / len(acc_list)
        train_a.append(epAcc)
        print('epoch {}    train/loss {}    train/accuracy {}    '.format(epoch, epLoss, epAcc), end="")

        # 1 epoch 終わったらテスト
        #+++++++++++++++++++++++++++++++++テスト　開始++++++++++++++++++++++++++++++++++++++
        test = np.array(test)
        testData = np.vstack(test[:, 0])
        testLabel = test[:, 1].astype(np.int32)

        testData = Variable(testData)
        testLabel = Variable(testLabel)

        # テストデータを入力
        testResult = model.forward(testData)

        # テスト誤差，テスト正解率を計算
        testLoss = np.mean(F.softmax_cross_entropy(testResult, testLabel).data)
        test_l.append(testLoss)
        testAcc = F.accuracy(testResult, testLabel).data
        test_a.append(testAcc)
        print('test/loss {}    test/accuracy {}'.format(testLoss, testAcc))
        #++++++++++++++++++++++++++++++++テスト　終了++++++++++++++++++++++++++++++++++++++++

        # epochをインクリメント
        epoch += 1

    elapsed = time.time() - start
    print(elapsed / 60)
    print(elapsed % 60)

    return max_epoch, train_l, train_a, test_l, test_a


def trainCNN(train, test):
    train_l = []
    train_a = []
    test_l = []
    test_a = []

    # モデルのインスタンス生成
    model = CNN(10)
    # 最適化手法の選択
    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)

    # ハイパーパラメータの設定
    epoch = 1
    max_epoch = 500
    batch_size = 1000
    start = time.time()

    gpuID = 0
    xp = cuda.cupy
    cuda.get_device(gpuID).use()
    model.to_gpu(gpuID)

    # max epoch　まで学習
    while epoch <= max_epoch:

        # epochごとに訓練データをシャッフル
        train = np.random.permutation(train)
        loss_list = []  # 誤差を入れる
        acc_list = []  # 正解率を入れる

        # ミニバッチごとに重みを更新,訓練データを使い切るまでループ
        # ---------------------------------------epoch 開始---------------------------------
        for i in range(0, len(train), batch_size):
            # 勾配を初期化
            model.cleargrads()

            # i番目からバッチサイズ分のデータを取ってくる
            # x:入力 t:ラベル
            x = xp.array([train[j, 0] for j in range(i, i + batch_size)])
            x = x[:, xp.newaxis, :, :]
            t = train[i:i + batch_size, 1]

            # chainerは基本Variable型に変換してからニューラルネットに入れる
            x = Variable(x)
            t = Variable(xp.array(t, dtype=xp.int32))

            # 出力層の値が返ってくる
            y = model.forward(x)

            # 正解率を計算
            acc = cuda.to_cpu(F.accuracy(y, t).data)
            acc_list.append(acc)

            # 誤差を計算 (出力をsoftmax関数に入れ，交差エントロピー誤差で誤差を算出)
            loss = F.softmax_cross_entropy(y, t)
            loss_list.append(np.sum(cuda.to_cpu(loss.data)))
            # 誤差から逆伝播して勾配を算出
            loss.backward()

            # 勾配からさっき選んだ最適化手法で重みを更新
            optimizer.update()

            print("{0:5d}/{1}".format(i+batch_size, len(train)))

        # --------------------------------epoch 終了---------------------------------------

        # 平均誤差
        epLoss = sum(loss_list) / len(train)
        train_l.append(epLoss)
        # 平均正解率
        epAcc = sum(acc_list) / len(acc_list)
        train_a.append(epAcc)
        print('epoch {}    train/loss {}    train/accuracy {}    '.format(epoch, epLoss, epAcc), end="")

        # 1 epoch 終わったらテスト
        # +++++++++++++++++++++++++++++++++テスト　開始+++++++++++++++++++++++++++++++++++++++
        test = np.array(test)
        testData = xp.array([data[0] for data in test])
        testData = testData[:, xp.newaxis, :, :]
        testLabel = test[:, 1]

        testData = Variable(testData)
        testLabel = Variable(xp.array(testLabel, dtype=xp.int32))

        # テストデータを入力
        testResult = model.forward(testData)

        # テスト誤差，テスト正解率を計算
        testLoss = np.mean(cuda.to_cpu(F.softmax_cross_entropy(testResult, testLabel).data))
        test_l.append(testLoss)
        testAcc = cuda.to_cpu(F.accuracy(testResult, testLabel).data)
        test_a.append(testAcc)
        print('test/loss {}    test/accuracy {}'.format(testLoss, testAcc))
        # ++++++++++++++++++++++++++++++++++テスト　終了++++++++++++++++++++++++++++++++++++++

        # epochをインクリメント
        epoch += 1

    elapsed = time.time() - start
    print(elapsed/60)
    print(elapsed%60)

    return max_epoch, train_l, train_a, test_l, test_a




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



# get dataset 1次元で取得
train_mlp, test_mlp = mnist.get_mnist(ndim=1, dtype=np.float32)
# get dataset 2次元で取得
train_cnn, test_cnn = mnist.get_mnist(ndim=2, dtype=np.float32)

#mnistを見てみる  0番目のデータの0個目に画像，1個目にラベルが入ってる
#plt.imshow(train_cnn[0][0], cmap='gray')
#print('shape: {}    label: {}'.format(train_cnn[0][0].shape, train_cnn[0][1]))
#plt.show()

#学習
max_epoch, train_l, train_a, test_l, test_a = trainMLP(train_mlp, test_mlp)
#max_epoch, train_l, train_a, test_l, test_a = trainCNN(train_cnn, test_cnn)

save_graph(max_epoch, train_l, train_a, test_l, test_a, 'tutorial_CNN_loss', 'tutorial_CNN_acc')