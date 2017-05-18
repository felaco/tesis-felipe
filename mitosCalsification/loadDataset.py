import tarfile

import cv2
import numpy as np
from PyQt5.QtCore import QDir

from common.Params import Params as P

imagesFilter = ['*.png', '*.tif', '*.bmp']

class dataset:
    """
    Instantiate to load the mitosis dataset. 
    Call the method get_training_sample to be able to use the dataset.
    """
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

    def get_training_sample(self, ratio = 1, shuffle=True, selection = True):
        """
        Create two arrays that can be used to train a model. The first one is the input for a model,
        the second one is the expected output
        :param ratio: The ratio between the amount of element of the mitosis class versus the no mitosis class
        :param shuffle: Choose true to shuffle the samples 
        :param selection: Select a sample of the no-mitosis or all of them
        :return: the input of a model and the expected output
        """
        if ratio <= 0:
            raise ValueError('ratio cannot be neither negative nor 0')

        selection_index = np.random.choice(len(self._cand_list),
                                           size=len(self._mitos_list) * ratio,
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
                im = self._resize(im)
                self._cand_list.append(self._preprocess(im))

    def _load_mitos(self):
        dir = QDir(self._mitos_folder)
        list = dir.entryInfoList(imagesFilter)

        for entry in list:
            path = entry.absoluteFilePath()
            im = cv2.imread(path)
            im = self._resize(im)
            self._mitos_list.append(self._preprocess(im))

    def _preprocess(self, im):
        im = im.astype(np.float32)
        im /= 255
        mean_dev = cv2.meanStdDev(im)
        mean = mean_dev[0]
        std_dev = mean_dev[1]
        im[:,:,0] -= mean[0]
        im[:,:,1] -= mean[1]
        im[:,:,2] -= mean[2]
        # im[:, :, 0] /= std_dev[0]
        # im[:, :, 1] /= std_dev[1]
        # im[:, :, 2] /= std_dev [2]
        return im

    def _resize(self, im):
        height = im.shape[0]
        if height == P().model_input_size:
            return im

        size = P().model_input_size
        return cv2.resize(im,(size,size), interpolation=cv2.INTER_CUBIC)



if __name__ == "__main__":
    train = dataset(P.Params().saveCutCandidatesDir + 'candidates.tar', P.Params().saveCutMitosDir)
    xe, ye = train.get_training_sample()
    test = dataset(P.Params().saveTestCandidates + 'candidates.tar', P.Params().saveTestMitos)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    print(xe.shape, ye.shape)
    print(xt.shape, yt.shape)
