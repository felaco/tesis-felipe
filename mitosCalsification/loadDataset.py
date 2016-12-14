from PyQt5.QtCore import QDir
import cv2
import numpy as np


def loadMitosDatasetTest():
    dirTest1 = QDir('C:/Users/home/Desktop/mitos dataset eval/test/mitosis')
    dirTest2 = QDir('C:/Users/home/Desktop/mitos dataset eval/test/noMitosis')
    dirTrain1 = QDir('C:/Users/home/Desktop/mitos dataset/train/mitosis')
    dirTrain2 = QDir('C:/Users/home/Desktop/mitos dataset/train/noMitos')
    dirVal = QDir('C:/Users/home/Desktop/mitos dataset/cutted/noMitosis')
    dirVal2 = QDir('C:/Users/home/Desktop/mitos dataset eval/cutted/noMitosis')

    imagesFilter = ['*.png', '*.tif', '*.bmp']

    mitosClassTestInfoList = dirTest1.entryInfoList(imagesFilter)
    noMitosClassTestInfoList = dirTest2.entryInfoList(imagesFilter)
    mitosClassTrainInfoList = dirTrain1.entryInfoList(imagesFilter)
    noMitosClassTrainInfoList = dirTrain2.entryInfoList(imagesFilter)
    validationInfoList = dirVal.entryInfoList(imagesFilter)
    valNoMitosInfoList = dirVal2.entryInfoList(imagesFilter)

    mitosImageList = []
    mitosNameList = []
    noMitosImageList = []
    noMitosNameList = []
    mitosTrainList = []
    noMitosTrainList = []
    hugeNoMitosList = []

    for entryInfo in noMitosClassTestInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        noMitosImageList.append(im)
        #basename = entryInfo.baseName()
        #noMitosNameList.append(path)

    for entryInfo in mitosClassTestInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        mitosImageList.append(im)
        basename = entryInfo.baseName()
        mitosNameList.append(basename)

    for entryInfo in mitosClassTrainInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        mitosTrainList.append(im)

    for entryInfo in noMitosClassTrainInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        noMitosTrainList.append(im)


    for entryInfo in valNoMitosInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        shape = im.shape
        if shape[0] != 63 or shape[1] != 63:
            print(path)
        im = im.reshape(3, 63, 63)
        im /= 255
        hugeNoMitosList.append(im)


    xe = np.append(mitosTrainList, noMitosTrainList, axis=0)
    ye = np.append(np.zeros(len(mitosTrainList)), np.ones(len(noMitosTrainList)))
    idx = np.arange(len(xe))
    np.random.shuffle(idx)
    xe = xe[idx]
    ye = ye[idx]

    xv = np.append(mitosImageList, noMitosImageList, axis=0)
    yv = np.append(np.zeros(len(mitosImageList)), np.ones(len(noMitosImageList)))
    idx = np.arange(len(xv))
    np.random.shuffle(idx)
    xv = xv[idx]
    yv = yv[idx]


    noMitosValImageList = []
    for entryInfo in validationInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        shape = im.shape
        if shape[0] != 63 or shape[1] != 63:
            print(path)
        im = im.reshape(3, 63, 63)
        im /= 255
        noMitosValImageList.append(im)
        noMitosNameList.append(path)
        i = 0

    retDict ={
        'xe':xe,
        'ye':ye,
        'xv':xv,
        'yv':yv,
        'noMitosTest':np.array(noMitosValImageList),
        'noMitosFile':noMitosNameList,
        'finalVal': np.array(hugeNoMitosList)
    }

    return retDict

ret = loadMitosDatasetTest()
i = 0