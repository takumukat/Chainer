import argparse
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os

import chainer

from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox


def level_correction(img, narrow):
    #narrow:階調をどれだけ狭めるか
    # ルックアップテーブルの生成
    min_table = narrow
    max_table = 255 - narrow
    print("max: {}   min: {}".format(max_table, min_table))
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


def histgram(img):
    r, g, b = img[0], img[1], img[2]

    hist_r, bins = np.histogram(r.ravel(), 256, [0, 256])
    hist_g, bins = np.histogram(g.ravel(), 256, [0, 256])
    hist_b, bins = np.histogram(b.ravel(), 256, [0, 256])

    plt.xlim(0, 255)
    plt.plot(hist_r, "-r", label="Red")
    plt.plot(hist_g, "-g", label="Green")
    plt.plot(hist_b, "-b", label="Blue")
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    #-----------------------------setting---------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained-model', default='voc0712')
    parser.add_argument('image')
    args = parser.parse_args()

    modelName = ['FasterRCNNVGG16', 'YOLOv2', 'YOLOv3', 'SSD300', 'SSD512']
    dataset = VOCBboxDataset(split='val', year='2012')
    print('len dataset: {}'.format(len(dataset)))

    folderList = os.listdir('tutorial_result/images')
    #---------------------------------------------------------------------
    #"""

    lim = 10
    x = [255.0 / (255.0 - n*5 * 2) for n in range(lim)]

    for num, name in enumerate(modelName):
        model = globals()[name](n_fg_class=len(voc_bbox_label_names), pretrained_model='voc0712')

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

        # read_image -> return (C, H, W) 3ch RGB float32 ndarray
        img = utils.read_image(args.image, color=True)

        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        #plt.figure(num)
        #vis_bbox(
        #    img, bbox, label, score, label_names=voc_bbox_label_names)
        #plt.savefig("tutorial_result/images/{}.png".format(name))



        dic = {'car1':[], 'car2':[], 'person1':[], 'person2':[]}
        for n in range(lim):
            h_img, l_img = level_correction(img, n*5)

            if not "tutorial_result/images/max{}min{}.jpg".format(255-n*5, n*5) in folderList:
                Image.fromarray(h_img.transpose(1, 2, 0)).save("tutorial_result/images/max{}min{}.jpg".format(255-n*5, n*5))

            h_img = np.float32(h_img)
            bboxes, labels, scores = model.predict([h_img])
            bbox, label, score = bboxes[0], labels[0], scores[0]
            print("{}:  {}   {}".format(name, label, score))

            if len(label) == 3:
                dic['car1'].append(0)
                dic['car2'].append(score[0])
                dic['person1'].append(score[1])
                dic['person2'].append(score[2])
            elif len(label) == 2:
                dic['car1'].append(0)
                dic['car2'].append(score[0])
                dic['person1'].append(0)
                dic['person2'].append(score[1])
            elif len(label) == 4:
                dic['car2'].append(score[0])
                dic['car1'].append(score[1])
                dic['person1'].append(score[2])
                dic['person2'].append(score[3])


        plt.figure(num)
        for k in sorted(dic):
            plt.plot(x, dic[k], '-o',label=k)
        plt.title("{}".format(name), fontsize=14)
        plt.xlabel("slope", fontsize=14)
        plt.ylabel("score", fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig("tutorial_result/{}.png".format(name))












    """
    #----------------------- find car and person image ----------------------
    model = SSD300(n_fg_class=len(voc_bbox_label_names), pretrained_model='voc0712')
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

    i = 0
    j = 0
    for t,data in enumerate(dataset):
        img = data[0]

        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        if voc_bbox_label_names.index('car') in label and voc_bbox_label_names.index('person') in label:
            #cv2.imwrite("../../carANDperson_12val/carANDperson{0:03d}.jpg".format(i), img[::-1].transpose(1,2,0))
            #print(i)
            if i == 117:
                print(t)
            i += 1


        elif voc_bbox_label_names.index('car') in data[2] and voc_bbox_label_names.index('person') in data[2]:
            #cv2.imwrite("../../carANDperson_12val_negative/carANDperson{0:03d}.jpg".format(j), img[::-1].transpose(1, 2, 0))
            j += 1

    print('all car and person: {}'.format(i+j))
    print('positive: {}'.format(i))
    print('negative: {}'.format(j))
    #vis_bbox(img, bbox, label, score, label_names=voc_bbox_label_names)
    #plt.show()
    #print(bbox)
    #print([voc_bbox_label_names[l] for l in label])
    #print(score)
    #"""




if __name__ == '__main__':
    main()
