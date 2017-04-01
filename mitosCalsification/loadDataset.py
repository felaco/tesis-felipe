import tarfile

import cv2
import numpy as np
from PyQt5.QtCore import QDir

from common.Params import Params as P

imagesFilter = ['*.png', '*.tif', '*.bmp']

def loadMitosDatasetTest():
    dirTest1 = QDir('C:/Users/home/Desktop/mitos dataset eval/test/mitosis')
    dirTest2 = QDir('C:/Users/home/Desktop/mitos dataset eval/test/noMitosis')
    dirTrain1 = QDir('C:/Users/home/Desktop/mitos dataset/train/mitosis')
    dirTrain2 = QDir('C:/Users/home/Desktop/mitos dataset/train/noMitos')
    dirVal = QDir('C:/Users/home/Desktop/mitos dataset/cutted/noMitosis')
    dirVal2 = QDir('C:/Users/home/Desktop/mitos dataset eval/cutted/noMitosis')


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
    print('No-mitos test loaded...')

    for entryInfo in mitosClassTestInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        mitosImageList.append(im)
        basename = entryInfo.baseName()
        mitosNameList.append(basename)

    print('mitos test loaded...')

    for entryInfo in mitosClassTrainInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        mitosTrainList.append(im)

    print('mitos train loaded...')

    for entryInfo in noMitosClassTrainInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        im = im.reshape(3, 63, 63)
        im /= 255
        noMitosTrainList.append(im)

    print('no-mitos train loaded...')


    for entryInfo in valNoMitosInfoList:
        path = entryInfo.absoluteFilePath()
        im = cv2.imread(path).astype(np.float32)
        shape = im.shape
        if shape[0] != 63 or shape[1] != 63:
            print(path)
        im = im.reshape(3, 63, 63)
        im /= 255
        hugeNoMitosList.append(im)

    print('No-mitos candidates loaded...')

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

    print('no mitos validation loaded...')

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

class dataset:
    def __init__(self, cand_file, mitos_folder):
        self._cand_file = cand_file
        self._mitos_folder = mitos_folder
        self._cand_list = []
        self._mitos_list = []
        print("Cargando candidatos...")
        self._load_candidates()
        print("Cargando Mitos...")
        if mitos_folder is not None:
            self._load_mitos()
            print("done...")

    def get_training_sample(self, shuffle=True, selection = True):
        selection_index = np.random.choice(len(self._cand_list),
                                           len(self._mitos_list),
                                           replace=False).astype(int)

        if selection:
            no_mitos_list = np.asarray(self._cand_list)[selection_index]
        else:
            no_mitos_list = np.asarray(self._cand_list)

        xe = np.append(self._mitos_list, no_mitos_list, axis=0)
        ye = np.append(np.zeros(len(self._mitos_list)),
                       np.ones(len(no_mitos_list))).astype(int)

        if shuffle:
            shuffle_index = np.arange(len(xe))
            np.random.shuffle(shuffle_index)
            xe = xe[shuffle_index]
            ye = ye[shuffle_index]

        return xe,ye

    def _load_candidates(self):
        tar = tarfile.TarFile(self._cand_file, "r")
        members = tar.getmembers()
        self.tar_file_names = tar.getnames()

        for mem in members:
            f = tar.extractfile(mem)
            if f is not None:
                a = f.read()
                encoded = np.frombuffer(a, dtype=np.uint8)
                im = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                self._cand_list.append(self._preprocess(im))

    def _load_mitos(self):
        dir = QDir(self._mitos_folder)
        list = dir.entryInfoList(imagesFilter)

        for entry in list:
            path = entry.absoluteFilePath()
            im = cv2.imread(path)
            self._mitos_list.append(self._preprocess(im))

    def _preprocess(self, im):
        im = im.astype(np.float32)
        im /= 255
        return im





if __name__ == "__main__":
    train = dataset(P.Params().saveCutCandidatesDir + 'candidates.tar', P.Params().saveCutMitosDir)
    xe, ye = train.get_training_sample()
    test = dataset(P.Params().saveTestCandidates + 'candidates.tar', P.Params().saveTestMitos)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    print(xe.shape, ye.shape)
    print(xt.shape, yt.shape)
