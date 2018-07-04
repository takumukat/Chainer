# -*- coding: utf-8 -*-
# making video dataset
# 動画の最後のフレームで正解，不正解を判断する

import os
import numpy as np
import glob
from PIL import Image
import cv2
import random

#root----
#       |--fall
#       |--lie
#       |--sitdown----
#                    |--2m
#                    |  .
#                    |  .
#                    |  .
#                    |--6m----
#                            |--video0
#                            |--video1
#                            |    .
#                            |    .
#                            |    .
#                            |--videox----
#                                        |--image0
#                                        |--image1
#                                        |    .
#                                        |    .
#                                        |    .
#                                        |--imagex

class MakeData(object):
    def __init__(self, root):
        # 動画のrootのパス
        self.root = root


    # 行動ごとにラベルを振り分ける
    # labelListがNoneなら0,1,2,...と順番にラベルをつける
    def makeLabel(self, path, labelList):
        if labelList is None:
            label = [i for i in range(len(os.listdir(path)))]
        else:
            label = labelList

        pathAndLabels = []

        dirs = sorted(os.listdir(path))
        for i in range(len(dirs)):
            pathAndLabels.append([path + dirs[i] + '/', np.int32(label[i])])
            print('{0}:  label {1}'.format(dirs[i], label[i]))

        return pathAndLabels


    # "訓練用のみ"，"テスト用のみ"，"訓練とテストに分ける" の3種類の属性を加える
    # attribute: 0 => train & test    1 => only train    2 => only test
    #            [0,2,0] => [train & test, only test, train & test]
    # attributeがNoneならすべてtrain&testになる
    def addAttribute(self, pathANDlabels, attribute):
        attrlist = ['train&test', 'only train', 'only test']
        if attribute is None:
            attr = [0] * len(os.listdir(pathANDlabels[0][0]))
        else:
            attr = attribute

        pANDls = pathANDlabels
        new_pANDls = []

        for pANDl in pANDls:
            dirs = sorted(os.listdir(pANDl[0]))
            for i, dir in enumerate(dirs):
                    new_pANDls.append([pANDl[0]+dir+'/', pANDl[1], attr[i]])
                    print('{}:{}'.format(dir, attrlist[attr[i]]))

        return new_pANDls


    # 訓練用，テスト用のパスとラベルをセットにしたリストを作る
    def makeTrainTest(self, new_pANDls, ratio):
        train = []
        test = []

        for pANDl in new_pANDls:
            videos = os.listdir(pANDl[0])
            if pANDl[2] == 1:
                for video in videos:
                    train.append([pANDl[0]+video+'/', pANDl[1]])

            elif pANDl[2] == 2:
                for video in videos:
                    test.append([pANDl[0]+video+'/', pANDl[1]])

            else:
                train_num = int(ratio * len(videos))
                for i in range(len(videos)):
                    if i < train_num:
                        train.append([pANDl[0]+videos[i]+'/', pANDl[1]])
                    else:
                        test.append([pANDl[0]+videos[i]+'/', pANDl[1]])

        random.shuffle(train)
        random.shuffle(test)

        return train, test


    # 動画をキーに,パスを値にした辞書を作成
    # 入力:パス
    # 出力：動画のバイトをキー，パスを値とした辞書
    def makeDict(self):
        pANDls = self.makeLabel(self.root, None)
        pANDls = self.addAttribute(pANDls, [1]*len(os.listdir(pANDls[0][0])))
        # _pANDlsにすべての動画のパスとラベルを入れる
        _pANDls, _ = self.makeTrainTest(pANDls, 1.0)

        videoDict = {}

        for pANDl in _pANDls:
            # 動画の全フレームのパスを取得
            try:
                imgNameList = sorted(glob.glob(pANDl[0] + '*'))
            except:
                continue

            imgSeq = []  # 動画を入れる配列

            # フレームのデータを順々入れていく
            for imgName in imgNameList:
                image = np.array(Image.open(imgName), dtype=np.float32)
                image = image / 255.0
                imgSeq.append(image)

            # numpy arrayに変換
            imgSeq = np.array(imgSeq, dtype=np.float32)
            # byteに変換してキーとし，動画のパスを値にする
            videoDict[imgSeq.tobytes()] = pANDl[0]

        return videoDict

        # ミニバッチ作成

    # 入力:画像1枚，最大輝度，最小輝度
    # 出力：ハイコントラスト，ローコントラストに変換した画像
    def contrast(self, img, min, max):
        # ルックアップテーブルの生成
        min_table = min
        max_table = max
        diff_table = max_table - min_table

        LUT_HC = np.arange(256, dtype='uint8')
        LUT_LC = np.arange(256, dtype='uint8')

        # ハイコントラストLUT作成
        for i in range(0, min_table):
            LUT_HC[i] = 0
        for i in range(min_table, max_table):
            LUT_HC[i] = 255 * (i - min_table) / diff_table
        for i in range(max_table, 255):
            LUT_HC[i] = 255

        # ローコントラストLUT作成
        for i in range(256):
            LUT_LC[i] = min_table + i * (diff_table) / 255

        # 変換
        src = np.array(img, dtype=np.uint8)
        high_cont_img = cv2.LUT(src, LUT_HC)
        low_cont_img = cv2.LUT(src, LUT_LC)

        return high_cont_img, low_cont_img


    def videoANDlabel(self, pathAndLabels):
        videoAndLabels = []

        for pANDl in pathAndLabels:
            # 動画1つを格納する配列
            Seq = []
            # 動画1つのラベルを格納する配列
            lossLabelSeq = []
            accLabelSeq = []

            #ソートした動画のフレームのパス
            seq = sorted(glob.glob(pANDl[0] + '*'))

            for frame in seq:
                image = Image.open(frame)
                img = np.array(image, dtype=np.float32)

                img = img / 255.0

                Seq.append(img)
                lossLabelSeq.append(pANDl[1])
                accLabelSeq.append(-1)

            # 最後以外は-1が入り，最後だけラベルを入れることで，最後のフレームだけで判断できる
            accLabelSeq[-1] = pANDl[1]

            videoAndLabels.append([np.array(Seq, dtype=np.float32), np.array(lossLabelSeq, dtype=np.int32), np.array(accLabelSeq, dtype=np.int32)])

        return videoAndLabels


    # rootのパスからデータセットの配列を作成
    def dataset(self, labelList, attribute, ratio):

        pANDls = self.makeLabel(self.root, labelList)
        pANDls = self.addAttribute(pANDls, attribute)
        train, test = self.makeTrainTest(pANDls, ratio)
        trainDS = self.videoANDlabel(train)
        testDS = self.videoANDlabel(test)

        return trainDS, testDS