import io
import json
import math
import sys
import tarfile

import cv2
from PyQt5.QtCore import QDir

from common import Params as P


# description of the parameters:
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

def createBlobDetector():
    params = P.Params().blobDetectorParams
    return cv2.SimpleBlobDetector_create(params)


def listFiles(folderPath, filters=[]):
    folder = QDir(folderPath)
    fileList = folder.entryInfoList(filters)
    return fileList


# snippet found in http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = ' ')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()

def print_progress_bar(progress, total):
    suffix = "{}/{} Completed".format(progress, total) # message for progress bar
    prefix = "Progress "
    printProgressBar(progress, total, prefix, suffix, length=50, fill='=')

class Coordinate:
    def __init__(self,x, y, img_base_name):
        self.x = x
        self.y = y
        self.img_base_name = img_base_name

    def to_dict(self):
        return {'col' : self.y, 'row': self.x}

def euclidianDistance(p1, p2):
    dy = p1[0] - p2[0]
    dx = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


class MitosVerification:
    COLUMN = 0
    ROW = 1

    def __init__(self):
        file = open(P.Params().mitosAnotationJsonPath)
        string = file.read()
        self.jsonDict = json.loads(string)
        file.close()
        self.verificated_mitos = 0
        self.base_name = None
        self.not_found_points = []

    def is_mitos(self, candidatePoint, minDist=25):
        mitosCenterList = self.get_mitos_points(self.base_name)

        for point in mitosCenterList:
            p = (point["col"], point["row"])
            dist = euclidianDistance(candidatePoint, p)
            if dist < minDist:
                self.verificated_mitos += 1
                try:
                    self.not_found_points.remove(point)
                except ValueError:
                    pass

                return True

        return False

    def set_base_name(self, base_name):
        self.base_name = base_name
        self.not_found_points = list(self.get_mitos_points(base_name))
        self.verificated_mitos = 0

    def get_mitos_count(self):
        return len(self.jsonDict[self.base_name])

    def get_mitos_points(self, base_name):
        return self.jsonDict[base_name]

    def print_verification_result(self):
        mitos_count = self.get_mitos_count()
        if self.verificated_mitos < mitos_count:
            print('\r{} {}/{}'.format(self.base_name,
                                    self.verificated_mitos,
                                    mitos_count))


if __name__ == "__main__":
    im = cv2.imread("C:/Users/home/a.png")
    _, buf = cv2.imencode(".png", im)
    string = buf.tostring()
    byt = bytearray(string)
    file = io.BytesIO(byt)
    tar = tarfile.TarFile("C:/Users/home/f.tar","a")
    info = tarfile.TarInfo("folder1/a.png")
    info.size = len(byt)

    tar.addfile(info, file)
    tar.close()
    i=0
