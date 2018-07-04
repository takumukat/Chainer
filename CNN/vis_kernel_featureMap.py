# -*- coding: utf-8 -*-
# convolutionのカーネルの可視化

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer import serializers

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os


class CNN(chainer.Chain):

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

        #convをかけた後の特徴マップを返す
        return h1, h4, h7


#これじゃダメ
def convImg(array):
    conv = array
    convTile = Image.new('F', (8 * conv.shape[2],  int(conv.shape[0]/8) * conv.shape[1]))

    z = 0
    for j in range(int(conv.shape[0]/8)):
        for i in range(8):
            convTile.paste(Image.fromarray(conv[z], mode='F'), (conv.shape[2] * i, conv.shape[1] * j))
            z += 1

    conv1Img = Image.fromarray(np.uint8(convTile))
    conv1Img.show('conv')
    plt.show(convTile)


#こっちのほうがきれいに出る
def vis_square(data):
    """形が (n, height, width) か (n, height, width, 3) の配列を受け取り
      (height, width) をおおよそ sqrt(n) by sqrt(n) のグリッドに表示"""

    # まずはデータを正規化
    data = (data - data.min()) / (data.max() - data.min())

    # 可視化するフィルタの数を正方形に揃える
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # フィルタの間に隙間を挿入
               + ((0, 0),) * (data.ndim - 3))  # 最後は隙間なし
    data = np.pad(data, padding, mode='constant', constant_values=0)  # 隙間は白

    # フィルタをタイル状に並べて画像化する
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
    plt.gray()
    plt.show()


def main():
    PATH = 'datasets/IRimage/'

    #画像を開く(80x60 gray image)
    #folder = os.listdir(PATH)
    imgList = glob.glob(PATH+'3m00/*')
    image = Image.open(imgList[2])
    #0~1に正規化
    image = Variable( (np.array(image, dtype=np.float32) / 255.0).reshape(1, 1, 60, 80) )

    #モデルに重みをロード
    model = CNN()
    serializers.load_npz('model/rightleft2400-3000_ep050.model', model)

    #w1 = model.conv1.W.data
    #w1 = w1.reshape(w1.shape[0], w1.shape[2], w1.shape[3])

    conv1, conv2, conv3 = model(image)

    # バッチサイズ=1固定だから中身の部分だけ取り出す
    conv1 = conv1.data.reshape(conv1.data.shape[1], conv1.data.shape[2], conv1.data.shape[3])
    conv2 = conv2.data.reshape(conv2.data.shape[1], conv2.data.shape[2], conv2.data.shape[3])
    conv3 = conv3.data.reshape(conv3.data.shape[1], conv3.data.shape[2], conv3.data.shape[3])

    print('conv1:{}'.format(conv1.shape))
    print('conv2:{}'.format(conv2.shape))
    print('conv3:{}'.format(conv3.shape))

    #convImg(conv2)
    vis_square(conv1)
    #vis_square(conv2)
    #vis_square(conv3)

    #vis_square(w1)


if __name__ == '__main__':
    main()