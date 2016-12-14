import cv2
import numpy as np

class Rectangulo:
    def __init__(self, left, top, width, height):
        self.left = int(left)
        self.top = int(top)
        self.right = int(left + width)
        self.bottom = int(top + height)

    def getCenter(self):
        w = self.right - self.left
        cx = int(self.left + (w/2))
        h = self.bottom - self.top
        cy = int(self.top + (h/2))
        return cx, cy

    def topLeft(self):
        return self.left, self.top

    def bottomRight(self):
        return self.right, self.bottom

    def checkIntegrity(self, maxX = 2084, maxY = 2084):
        if self.left < 0 or self.top < 0 or self.bottom > maxY or self.right > maxX:
            return False
        return True

class CandidateSelection:
    def __init__(self, nameImage, mitosRect, winsize=63):
        self.nameImage = nameImage
        self.mitosRect = mitosRect
        self.candPath = 'C:/Users/home/Desktop/mitos dataset/candidatos/'
        self.winSize = winsize

    def selectCandidates(self):
        self.image = cv2.imread(self.candPath + self.nameImage, cv2.IMREAD_GRAYSCALE)

        '''
        im2 = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for c in self.mitosRect:
            cv2.rectangle(im2, c.topLeft(), c.bottomRight(),(0,0,255), 2)
        cv2.imwrite('holapCand.bmp', im2)
        '''
        _, points, _ = cv2.findContours(self.image,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

        rects = self.createRectList(points)
        self.filterByDistance(rects)
        return rects

    def createRectList(self, points):
        list = []
        for point in points:
            x, y, w, h = cv2.boundingRect(point)
            cx = int (x + w / 2)
            cy = int (y + h / 2)
            rect = Rectangulo(cx - self.winSize/2,
                              cy - self.winSize/2,
                              self.winSize,
                              self.winSize)
            list.append(rect)

        return list

    def filterByDistance(self, rois):
        i = 0
        #for rects in rois:
        while i < len(rois):

            if(self.detectIntersections(rois[i], self.mitosRect)):
                del rois[i]
                continue

            if(self.detectCloseRegions(rois[i], rois, i)):
                del rois[i]
                continue
            i += 1
            #print('%d / %d' %(i, len(rois)))




    def isIntersected(self, rect1, rect2):
        if(rect1.left < rect2.right and
           rect1.right > rect2.left and
           rect1.top < rect2.bottom and
           rect1.bottom > rect2.top):

            return True
        return False

    def detectIntersections(self, rect, list):
        for l in list:
            if self.isIntersected(rect, l):
                return True
        return False

    def isClose(self, cx1, cy1, cx2, cy2):
        winsize2 = (self.winSize*2/3) * (self.winSize*2/3)
        dx2 = (cx2-cx1)*(cx2-cx1)
        dy2 = (cy2-cy1)*(cy2-cy1)
        if dx2 + dy2 < winsize2:
            return True
        return False

    def detectCloseRegions(self, rect, list, pos):
        cx1, cy1 = rect.getCenter()
        i = pos + 1
        count = 0
        flag = False
        while i < len(list) - count:
            cx2, cy2 = list[i].getCenter()
            if self.isClose(cx1, cy1, cx2, cy2):
                list.append(self.fuseRect(list[i], rect))
                del list[i]
                count += 1
                flag = True
                continue
            i += 1
        return flag

    def fuseRect(self, rect1, rect2, winsize=63):
        cx1, cy1 = rect1.getCenter()
        cx2, cy2 = rect2.getCenter()
        x = int ((cx1 + cx2) / 2)
        y = int ((cy1 + cy2) / 2)
        newRect = Rectangulo(int(x - winsize/2), int(y - winsize/2), winsize, winsize)
        return newRect



if __name__ == "__main__":
    array  = [1, 2, 3, 4, 3,2, 5]


    def filter(array, num, pos):
        i = pos + 1
        while i < len(array):
            p = array[i]
            if p == num:
                #del array[i]
                return True
            i += 1
        return False

    i= 0
    while i < len(array):
        p = array[i]
        if(filter(array, p, i)):
            del array[i]
            continue
        i += 1
        print(array)



    im = cv2.imread('C:/Users/home/Desktop/mitos dataset/candidatos/A00_01.bmp',
                    cv2.IMREAD_GRAYSCALE)
    c = CandidateSelection(im, None)
    _,points,_ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list = c.createRectList(points)
    c.filterByDistance(list)
    im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for p in list:
        cv2.rectangle(im2, (p.left, p.top), (p.right, p.bottom), (0,0,255), 2)

    cv2.imwrite('holap.png', im2)

