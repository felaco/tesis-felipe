from PyQt5.QtCore import QDir
import candidateSelection as cs
import os
import numpy as np
import cv2


def debugAnotations(im, centroids, winsize = 63):
    im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for c in centroids:
        cx = c[0]
        cy = c[1]
        x = cx - (int(winsize / 2))
        y = cy - (int(winsize / 2))
        cv2.rectangle(im2, (x,y), (x+winsize, y+winsize), (0,0,255), 3)

    cv2.imwrite('holap.png', im2)

def generatePoint (splittedString):
    i = 0
    pointsList = []
    while i < len(splittedString):
        x = splittedString[i]
        y = splittedString[i+1]
        pointsList.append([x,y])
        i += 2

    return np.array(pointsList).astype(np.int32)

def paintAnotations(points):
    im = np.zeros((2084, 2084), np.uint8)
    for point in points:
        x = point[:,0]
        y = point[:,1]
        im[y,x] = 255

    #cv2.imwrite('holap.png', im)
    return im

def findCentroids (contours):
    centroids = []
    for c in contours:
        moments = cv2.moments(c)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centroids.append([cx, cy])

    return centroids

def findCentroidsByBounRect(points):
    centroids = []
    for point in points:
        x,y,w,h = cv2.boundingRect(point)
        cx = int(x + (w/2))
        cy = int(y + (h/2))
        centroids.append([cx, cy])

    return centroids

def saveCuttedImage(image, centroids, winSize=63, dirPath='', baseName=''):
    i = 1

    if centroids[0].__class__.__name__  == 'Rectangulo':
        for r in centroids: #lista de rectangulos
            if not r.checkIntegrity():
                continue
            savePath = dirPath + '/' + baseName + '-' + 'NonMitos-' + str(i) + '.png'
            cut = image[r.top: r.bottom, r.left: r.right]
            cv2.imwrite(savePath, cut)
            i += 1
            if i == 279:
                i = i
        return

    for c in centroids:
        cx = c[0]
        cy = c[1]
        x = cx - (int(winSize/2))

        if x < 0:
            x = 0
        elif x + winSize > image.shape[1]:
            x = image.shape[1] - winSize
        y = cy - (int(winSize/2))

        if y < 0:
            y = 0
        elif y + winSize > image.shape[0]:
            y = image.shape[0] - winSize

        savePath = dirPath + '/' + baseName + '-' + str(i) + '.png'
        mitos = image[y: y+winSize, x: x+winSize]
        cv2.imwrite(savePath, mitos)
        i += 1

def getFileList(baseDir):
    anotationsFilter = ['*.csv']
    imagesFilter = ['*.png', '*.tif', '*.bmp']
    anotationsFilter = ['A02_02.csv']
    imagesFilter = ['A02_02.bmp']
    anotationsFileList = baseDir.entryInfoList(anotationsFilter)
    imagesFileList = baseDir.entryInfoList(imagesFilter)
    return anotationsFileList, imagesFileList

def createBRectList(centroids, winsize=63):
    rectList = []
    for c in centroids:
        cx = c[0]
        cy = c[1]
        x = cx - (int(winsize / 2))
        y = cy - (int(winsize / 2))
        rect = cs.Rectangulo(x, y, winsize, winsize)
        rectList.append(rect)
    return rectList



baseDir = QDir('C:/Users/home/Desktop/mitos dataset')
saveDir = QDir(baseDir.absolutePath()+'/cutted/mitosis')
saveDirNonMitos = QDir(baseDir.absolutePath()+'/cutted/noMitosis')

anotationsFileList, imagesFileList = getFileList(baseDir)

i = 0

while i < len(anotationsFileList):
    anotationAbsPath = anotationsFileList[i].absoluteFilePath()
    imageAbsPath = imagesFileList[i].absoluteFilePath()
    anotationCsv = open(anotationAbsPath)
    mitosRegion = []
    for line in anotationCsv:
        splitted = str.split(line, ',')
        point = generatePoint(splitted)
        mitosRegion.append(point)

    #centroids = findCentroids(cont)
    centroids = findCentroidsByBounRect(mitosRegion)
    im = cv2.imread(imageAbsPath)

    basenameExt = os.path.basename(imageAbsPath)
    baseName, _ = os.path.splitext(basenameExt)


    rectList = createBRectList(centroids)
    candSelector = cs.CandidateSelection(basenameExt, rectList)
    rectList = candSelector.selectCandidates()


    #saveCuttedImage(image=im,
     #               centroids=rectList,
      #              dirPath=saveDirNonMitos.absolutePath(),
       #             baseName=baseName)



    im2 = cv2.imread('C:/Users/home/Desktop/mitos dataset/candidatos/A02_02.bmp')
    for p in rectList:
        cv2.rectangle(im2, (p.left, p.top), (p.right, p.bottom), (0,0,255), 2)

    cv2.imwrite('holap.png', im2)



    #saveCuttedImage(image= im,centroids=centroids, dirPath=saveDir.absolutePath(), baseName=baseName)
    #debugAnotations(binaryMitos,centroids)
    i += 1
    print('%d / %d' % (i, len(anotationsFileList)))


