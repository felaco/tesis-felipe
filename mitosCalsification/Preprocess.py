from PyQt5.QtCore import QDir
import cv2
import os


def rotateImageAndSave(image, baseName, path):
    k = 1
    rows, cols, _ = image.shape
    while k <= 3:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90*k, 1)
        res = cv2.warpAffine(image, M, (cols, rows))
        name = path+baseName+'-rot-'+str(90*k)+'.png'
        cv2.imwrite(name, res)
        k += 1


#dirTrain1 = QDir('C:/Users/home/Desktop/mitos dataset/train/mitosis')
dirTrain2 = QDir('C:/Users/home/Desktop/mitos dataset/train/noMitos')
imagesFilter = ['*.png', '*.tif', '*.bmp']

#info1 = dirTrain1.entryInfoList(imagesFilter)
info2 = dirTrain2.entryInfoList(imagesFilter)
infoList = []
#infoList.extend(info1)
infoList.extend(info2)

i = 0
for fileInfo in infoList:
    i += 1
    imagePath = fileInfo.absoluteFilePath()
    basePath = fileInfo.absolutePath() +'/'
    basenameExt = os.path.basename(imagePath)
    baseName, _ = os.path.splitext(basenameExt)

    imbase = cv2.imread(imagePath)
    rows, cols, _ = imbase.shape
    imXMirror = cv2.flip(imbase, 1)

    cv2.imwrite(basePath+baseName+'-mirror.png', imXMirror)
    rotateImageAndSave(imbase, baseName, basePath)
    rotateImageAndSave(imXMirror, baseName+'-mirror', basePath)

    print('%d / %d'%(i, len(infoList)))
