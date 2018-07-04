# coding: utf-8
# データセットを作る ver1

import os
import numpy as np
import glob
from PIL import Image


class MakeData(object):
    def __init__(self, gpuID):
        self.gpuID = gpuID

    # 動画フォルダのパスをシャッフル，行動ごとにラベル付け
    def makePath(self, path, ratio):
        dirs = sorted(os.listdir(path))

        pathsAndLabels = []
        trainPathAndLabels = []
        testPathAndLabels = []
        print('ラベル: {}'.format(dirs))

        # ディレクトリ上から順にラベル0,1,...
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
            train_num = int(ratio * len(videoPath))
            print('train: {0}   test: {1}'.format(train_num, len(videoPath)-train_num) )

            for i in range(train_num):
                trainPathAndLabels.append(np.asarray([pathAndLabel[0] + videoPath[i] + '/', pathAndLabel[1]]))

            for i in range(train_num, len(videoPath)):
                testPathAndLabels.append(np.asarray([pathAndLabel[0] + videoPath[i] + '/', pathAndLabel[1]]))

        return trainPathAndLabels, testPathAndLabels, dirs


    #動画をキーにパスを値にした辞書を作成
    def makeDict(self, testPath):

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

        # ミニバッチ作成



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

#ハイコントラスト，ローコントラスト画像もいっしょに作る
    def makeTrainData(self, imgNameList, label, videoAndLabels):

        # 動画を格納する配列
        origSeq = []
        highSeq = []
        lowSeq = []
        # ラベルを格納する配列
        orig_losslabelSeq = []
        high_losslabelSeq = []
        low_losslabelSeq = []
        orig_acclabelSeq = []
        high_acclabelSeq = []
        low_acclabelSeq = []

        #
        for imgName in imgNameList:
            image = Image.open(imgName)
            orig = np.array(image, dtype=np.float32)

            high, low = self.contrast(image, 30, 225)
            high = np.array(high, dtype=np.float32)
            low = np.array(low, dtype=np.float32)

            orig = orig / 255.0
            high = high / 255.0
            low = low / 255.0

            origSeq.append(orig)
            highSeq.append(high)
            lowSeq.append(low)

            orig_losslabelSeq.append(label)
            high_losslabelSeq.append(label)
            low_losslabelSeq.append(label)

            orig_acclabelSeq.append(-1)
            high_acclabelSeq.append(-1)
            low_acclabelSeq.append(-1)

        # 最後以外は-1が入り，最後だけラベルを入れることで，最後のフレームだけで判断できる
        orig_acclabelSeq[-1] = label
        high_acclabelSeq[-1] = label
        low_acclabelSeq[-1] = label


        # [video, loss label, acc label]のリストをアペンド
        origVideo = np.array(origSeq, dtype=np.float32)
        origLossLabel = np.array(orig_losslabelSeq, dtype=np.int32)
        origAccLabel = np.array(orig_acclabelSeq, dtype=np.int32)
        origVideoAndLabel = [origVideo, origLossLabel, origAccLabel]

        highVideo = np.array(highSeq, dtype=np.float32)
        highLossLabel = np.array(high_losslabelSeq, dtype=np.int32)
        highAccLabel = np.array(high_acclabelSeq, dtype=np.int32)
        highVideoAndLabel = [highVideo, highLossLabel, highAccLabel]

        lowVideo = np.array(lowSeq, dtype=np.float32)
        lowLossLabel = np.array(low_losslabelSeq, dtype=np.int32)
        lowAccLabel = np.array(low_acclabelSeq, dtype=np.int32)
        lowVideoAndLabel = [lowVideo, lowLossLabel, lowAccLabel]

        videoAndLabels.append(origVideoAndLabel)
        videoAndLabels.append(highVideoAndLabel)
        videoAndLabels.append(lowVideoAndLabel)

        return videoAndLabels


    def makeTestData(self, imgNameList, label, videoAndLabels):

        # 動画を格納する配列
        origSeq = []
        # ラベルを格納する配列
        orig_losslabelSeq = []
        orig_acclabelSeq = []

        #
        for imgName in imgNameList:
            image = Image.open(imgName)
            orig = np.array(image, dtype=np.float32)

            orig = orig / 255.0

            origSeq.append(orig)

            orig_losslabelSeq.append(label)

            orig_acclabelSeq.append(-1)

        # 最後以外は-1が入り，最後だけラベルを入れることで，最後のフレームだけで判断できる
        orig_acclabelSeq[-1] = label

        # [video, loss label, acc label]のリストをアペンド
        origVideo = np.array(origSeq, dtype=np.float32)
        origLossLabel = np.array(orig_losslabelSeq, dtype=np.int32)
        origAccLabel = np.array(orig_acclabelSeq, dtype=np.int32)
        origVideoAndLabel = [origVideo, origLossLabel, origAccLabel]

        videoAndLabels.append(origVideoAndLabel)

        return videoAndLabels


    #データセットの配列を作成
    def dataset(self, pathAndLabels, train):
        videoAndLabels = []

        #(batchsize)個の動画を１つのミニバッチにする
        for pathAndLabel in pathAndLabels:
            #各フレームのファイル名を取得
            imgNameList = sorted(glob.glob(pathAndLabel[0]+'*'))

            if train == True:
                videoAndLabels = self.makeTrainData(imgNameList, pathAndLabel[1], videoAndLabels)
            else:
                videoAndLabels = self.makeTestData(imgNameList, pathAndLabel[1], videoAndLabels)

        """
        #テストデータには4mの画像も入れる
        if train == False:
            _, te = self.makePath('/home/hashimoto/PycharmProjects/AI/datasets/fall_detection_4m/', ratio=0)
            for pAndL in te:
                imgNameList = sorted(glob.glob(pAndL[0]+'*'))
                videoAndLabels = self.makeTestData(imgNameList, pAndL[1], videoAndLabels)
        """
        print(len(videoAndLabels[0]))
        VandL = videoAndLabels if len(videoAndLabels) == 1 else np.random.permutation(np.array(videoAndLabels))

        return VandL
