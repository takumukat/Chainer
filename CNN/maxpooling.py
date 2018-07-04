# coding: utf-8
#画像にpoolingしたらどうなるか見てみる(gray scale image)

from PIL import Image
import chainer.functions as F
import numpy as np
import glob
import os

#max poolingを5回かける
def maxPooling(imgList, savePath, imgName):
    image = imgList
    canvas = Image.new('L', (80*5, 60*3))
    for j in range(len(image)):
        for i in range(4):
            print(image[j].shape)
            image[j] = F.max_pooling_2d(image[j], ksize=3, stride=2)
            pilImg = Image.fromarray(np.uint8(image[j].data.reshape(image[j].shape[2], image[j].shape[3])))
            canvas.paste(pilImg, (80*i, 60*j))
            #pilImg.save(savePath+'\\{}_{}.png'.format(imgName[j][:-4], i+1), 'PNG')
    #canvas.save(savePath+'\\maxpooling_{}_ksize2.png'.format(savePath[-4:]), 'PNG')

#maxとaverageを混ぜながら3回プーリングする(max,max,max -> max,max,average -> ... -> average,average,average)
def max_avePooling(imgList, savePath, imgName):
    canvas = Image.new('L', (80 * 8, 60 * 3))
    for k in range(len(imgList)):
        for j in range(8):          #j: 000~111の3ビット
            img = imgList[k]
            label = []
            jj = j  #シフトする用
            for i in range(3):
                if jj & 4 == 4:
                    img = F.max_pooling_2d(img, ksize=3, stride=2)
                    label.append('M')
                else:
                    img = F.average_pooling_2d(img, ksize=3, stride=2)
                    label.append('A')

                jj = jj << 1
            pilImg = Image.fromarray(np.uint8(img.data.reshape(img.shape[2], img.shape[3])))
            canvas.paste(pilImg, (80 * j, 60 * k))
            #pilImg.save(savePath + '\\{0}_{1}{2}{3}.png'.format(imgName[k][:-4], label[0],label[1],label[2]), 'PNG')
    #canvas.save(savePath+'\\max_ave_pooling_{}_ksize3.png'.format(savePath[-4:]), 'PNG')



def main():
    inputPath = 'datasets/IRimage/' #画像のフォルダが入ってる親フォルダのパス
    outputPath1 = 'datasets/pooling_test/max_result/stride2/'
    outputPath2 = 'datasets/pooling_test/max_ave_result/stride2/'
    folders = os.listdir(inputPath)

    #フォルダをひとつずつ
    for folder in folders:
        imgList = []
        #os.mkdir(outputPath1+folder)
        #os.mkdir(outputPath2+folder)
        imgName = os.listdir(inputPath+folder)
        print('{}:{}'.format(folder,imgName))
        imgPath = glob.glob(inputPath + folder + '/*')

        #画像をひとつずつ4次元テンソルに直す
        for path in imgPath:
            img = Image.open(path)
            imgList.append(np.array(img, dtype=np.float32).reshape(1, 1, 60, 80))

        imgList2 = np.copy(imgList)
        maxPooling(imgList, outputPath1+folder, imgName)

        #max_avePooling(imgList2, outputPath2+folder, imgName)


if __name__ == '__main__':
    main()