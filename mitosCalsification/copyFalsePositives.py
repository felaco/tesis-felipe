from PyQt5.QtCore import QFileInfo, QFile
import numpy as np

def copyFalsePositives(fileNames, prediction):
    target = np.ones(len(prediction))
    savedir = 'C:/Users/home/Desktop/mitos dataset/cutted/pasos/5/'

    i = 0
    while i < len(prediction):
        p = prediction[i]
        if p != target[i]:
            file = QFileInfo(fileNames[i])
            saveFilePath = savedir + file.fileName()
            QFile.copy(fileNames[i],saveFilePath)
            QFile.remove(fileNames[i])

        i += 1
        print('%d/%d',(i, len(prediction)))
