#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
"""
re = [ 'C:\\tool\\python\\python36.zip', 'C:\\tool\\python\\DLLs', 'C:\\tool\\python\\lib', 'C:\\tool\\python', 'C:\\tool\\python\\lib\\site-packages']
ap = ['C:\\tool\\ANACONDA','C:\\tool\\ANACONDA\\Lib\\site-packages','C:\\tool\\ANACONDA\\Lib', 'C:\\tool\\ANACONDA\\DLLs']

#for r in re:
#    sys.path.remove(r)
for a in ap:
    sys.path.append(a)
"""
print(sys.path)

#sys.exit()




import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from CNN_model import CNN
from chainer import serializers
from chainer import Variable
import chainer.functions as F


def main(gpuID, Path, label):
    category_label = label
    parent, img_name = os.path.split(Path)
    save_name = "..\\grad_cam\\test\\{}".format(img_name)

    #raw_img = Image.open(Path)
    raw_img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print('raw img: {}'.format(raw_img.shape))

    img = raw_img[np.newaxis][np.newaxis]
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    print('array: {}'.format(img.shape))

    model = CNN()
    serializers.load_npz('..\\model\\rightleft3m2400-3000_ep050.model', model)

    input_img = Variable(img)
    pred = model(input_img)          #推論結果
    probs = F.softmax(pred).data[0]  #softmaxの確率
    top2 = np.argsort(probs)[::-1]   #確率の高いTOP2のラベル
    print(top2)
    pred.zerograd()                  #勾配を初期化
    pred.grad = np.zeros([1, 2], dtype=np.float32)  #クラス数だけ0で初期化した配列を作る
    pred.grad[0, top2[category_label]] = 1          #選んだクラスのところだけ1にする

    words = np.array(['FALL', 'NOT FALL'])

    #TOP2のラベルと確率を表示
    #どのラベルの"推論結果に寄与した部分"を見たいかを表示(your choice)
    probs = np.sort(probs)[::-1]
    for w, p in zip(words[top2], probs):
        print('{0}   prob:{1:.05f}'.format(w, p))
    print("your choice ", words[top2[category_label]])
    pred.backward(True)     #逆伝播して勾配を計算

    # あらかじめモデルのclass内にcamという変数を作っておき，そこに注目している特徴マップを入れておく
    feature, grad = model.cam.data[0], model.cam.grad[0]
    cam = np.ones(feature.shape[1:], dtype=np.float32)   #後で割り算するからゼロにならないように1で初期化
    weights = grad.mean((1, 2)) * 1000  #特徴マップのチャネルごとの勾配の平均値を1000倍してる
    for i, w in enumerate(weights):  #camに特徴マップの各チャネルに対応する重み(weights)を掛け合わせ，加算していく
        cam += feature[i] * w
    cam = cv2.resize(cam, (80, 60))  #元の画像の大きさに伸ばす
    cam = np.maximum(cam, 0)         #0以下を0にする
    heatmap = cam / np.max(cam)      #camの中の最大値で割る

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)   #カラーにする

    fig = plt.figure(figsize=(10, 8))

    ax1 = plt.subplot()
    ax1.imshow(heatmap[:, :, ::-1])
    fig.savefig(save_name)



if __name__ == '__main__':
    Path = 'C:\\Users\\takum\\PycharmProjects\\AI\\dataset\\2018_0108\\17_12_25\\1225fall\\2m\\00\\'
    #Path = 'C:\\Users\\Takumi\\PycharmProjects\\OWLIFT\\image_data\\pooling_test\\IRimage\\left\\'

    #0:left 1:right
    label = 0

    folders = os.listdir(Path)
    for folder in folders:
        path = Path + folder + '\\' + os.listdir(Path+folder)[1]
        print(path)

        #main(-1, path, label)