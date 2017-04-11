import cv2
from sys import platform

class Params:
    _instance = None
    # Singleton class
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.blobDetectorParams = cv2.SimpleBlobDetector_Params()

        self.blobDetectorParams.filterByArea = True
        self.blobDetectorParams.minArea = 120
        self.blobDetectorParams.maxArea = 2000

        self.blobDetectorParams.filterByColor = False

        self.blobDetectorParams.minConvexity = 0.3
        self.blobDetectorParams.minInertiaRatio = 0.03
        self.blobDetectorParams.minDistBetweenBlobs = 20

        self.blobDetectorParams.minThreshold = 30
        self.blobDetectorParams.maxThreshold = 130

        self.blobDetectorParams.thresholdStep = 25

        if platform == 'win32':
            self.normHeStainDir = "C:/Users/felipe/mitos dataset/normalizado/heStain/"
            self.saveCandidatesWholeImgDir = "C:/Users/felipe/mitos dataset/train/print/"
            self.saveCutCandidatesDir = "C:/Users/felipe/mitos dataset/train/candidates/"
            self.saveMitosisPreProcessed = "C:/Users/felipe/mitos dataset/train/mitosis/preProcessed/"
            self.saveCutMitosDir = "C:/Users/felipe/mitos dataset/train/mitosis/"
            self.saveTestCandidates = "C:/Users/felipe/mitos dataset/test/"
            self.saveTestMitos = "C:/Users/felipe/mitos dataset/test/mitosis/"
            self.candidatesTrainingJsonPath = "C:/Users/felipe/mitos dataset/anotations/trainCandidates.json"
            self.candidatesTestJsonPath = "C:/Users/felipe/mitos dataset/anotations/testCandidates.json"
            self.mitosAnotationJsonPath = "C:/Users/felipe/mitos dataset/anotations/MitosAnotations.json"
        elif platform == 'linux':
            self.normHeStainDir = "/home/facosta/dataset/normalizado/heStain/"
            self.saveCandidatesWholeImgDir = "/home/facosta/dataset/train/print/"
            self.saveCutCandidatesDir = "/home/facosta/dataset/train/candidates/"
            self.saveMitosisPreProcessed = "/home/facosta/dataset/train/mitosis/preProcessed/"
            self.saveCutMitosDir = "/home/facosta/dataset/train/mitosis/"
            self.saveTestCandidates = "/home/facosta/dataset/test/"
            self.saveTestMitos = "/home/facosta/dataset/test/mitosis/"
            self.candidatesTrainingJsonPath = "/home/facosta/dataset/anotations/trainCandidates.json"
            self.candidatesTestJsonPath = "/home/facosta/dataset/anotations/testCandidates.json"
            self.mitosAnotationJsonPath = "/home/facosta/dataset/anotations/MitosAnotations.json"


Params().saveTestMitos = False
Params().saveMitosisPreProcessed = False
i=0