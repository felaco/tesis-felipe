from PyQt5 import QtWidgets as wid
from PyQt5.QtCore import QDir,  QFileInfo, pyqtSignal, QDir, QDirIterator
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import os
import cvTemplateMatch as tm

from templateMatcher import Ui_MainWindow
import numpy
import cv2

import threading

def getFileNameFromPath(path):
    basename = os.path.basename(path)
    filename,_ = os.path.splitext(basename)
    return filename

class MainWindow(Ui_MainWindow, wid.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.lineEditImage.setText('C:/Windows.old/Users/home/mitos_dataset')
        self.lineEditTemplate.setText('C:/Windows.old/Users/home/Desktop/New folder')
        self.ButtonTemplateSearch.clicked.connect(self.search_template)
        self.ButtonSearchImage.clicked.connect(self.search_image)
        self.ButtonAccept.clicked.connect(self.run)
        self.templateProgressIncreased.connect(self.onTemplateProgressIncreased)
        self.baseImageChanged.connect(self.onBaseImageProgressIncreased)
        self.baseImageChanged.connect(self.printImage)
        self.templateFoundInImage.connect(self.paint_image_with_template)
        self.templateFoundInImage.connect(self.setImageFocus)
        #self.templateFoundInImage.connect(self.saveTemplate)
        #self.changeStatusBarMessage.connect(self.onChangeStatusBarMessage)
        self.numpy_image = 0
        self.templateCount = 0
        self.workingImagePath = ""
        self.block = False

    def run(self):
        if self.block:
            return
        self.block = True
        tread = threading.Thread(target=self.iterateOnImages,
                                 name='Iterate on Images')
        tread.start()

    def iterateOnImages(self):
        image_path = self.lineEditImage.text()
        # self.find_template_in_image(image)
        directory = QDir(self.lineEditTemplate.text())
        filters = ['*.png', '*.tif', '*.jpg', '*.bmp']
        templateDirectoryList = directory.entryInfoList(filters)
        imageDirectoryList = self.get_image_path_list(image_path)


        i = 0
        while i < len(imageDirectoryList):
            self.workingImagePath = imageDirectoryList[i]
            progress = int(i / len(imageDirectoryList) * 100)
            image = cv2.imread(filename=self.workingImagePath,
                               flags=cv2.IMREAD_COLOR)
            self.numpy_image = image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.baseImageChanged.emit(progress)
            self.find_template_in_image(image,templateDirectoryList)
            i += 1

        self.saveRemainingTemplateList(templateDirectoryList)
        self.block = False

    def find_template_in_image(self, image, directoryList):

        if not self.lineEditImage.text() or not self.lineEditTemplate.text():
            return

        # iterator = QDirIterator(self.lineEditTemplate.text())

        i = 0
        val = 0
        self.templateCount = 0
        # while iterator.hasNext():
        for fileInfo in directoryList:
            # template_image_path = iterator.next()

            templatePath = fileInfo.absoluteFilePath()
            i += 1
            temp = i / len(directoryList) * 100

            if int(temp) != val:
                val = int(temp)
                self.templateProgressIncreased.emit(val)


            # val = i / len(directoryList) * 100
            # self.progressBarTemplate.setValue(int(val))
            temp_img = cv2.imread(filename=templatePath,
                                  flags=cv2.IMREAD_GRAYSCALE)

            found, max_pos, max_value = tm.run_template_match(image, temp_img)
            print('progreso: ' + str(val) + ' file: ' + fileInfo.fileName()
                  + ' res: ' + str(max_value))
            if found:
                message = 'Coincidencia encontrada con el temlate : ' + templatePath + \
                          ' en la posición: ' + str(max_pos[0]) + ' , ' + str(max_pos[1])
                #self.statusbar.showMessage(message)
                directoryList.pop(i)
                #self.changeStatusBarMessage.emit(message)
                templateName = getFileNameFromPath(templatePath)
                self.templateFoundInImage.emit(max_pos[0], max_pos[1])
                self.saveTemplate(max_pos[0], max_pos[1], image, templateName)
                #return

    def printImage(self):
        image = self.numpy_image_2_qimage(self.numpy_image)
        pixmap = QPixmap.fromImage(image)
        self.Imagen.setPixmap(pixmap)



    def paint_image_with_template(self, x, y):
        image = self.Imagen.pixmap()
        painter = QPainter(image)
        color = QColor(0, 102, 204)
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(x, y, 128, 128)
        self.Imagen.setPixmap(image)

    def search_template(self):
        fileName = wid.QFileDialog.getExistingDirectory(parent=self,
                                                        caption='Buscar Imagen Modelo',
                                                        directory=QDir.currentPath(),
                                                        options=wid.QFileDialog.ShowDirsOnly)

        if fileName:
            self.lineEditTemplate.setText(fileName)

    def search_image(self):
        file_name = wid.QFileDialog.getExistingDirectory(parent=self,
                                                         caption='Buscar Imagen principal',
                                                         directory=QDir.currentPath(),
                                                         options=wid.QFileDialog.ShowDirsOnly)

        if file_name:
            self.lineEditImage.setText(file_name)

    def get_image_path_list(self, basePath):
        filters = ['*.png', '*.tif', '*.jpg', '*.bmp']
        fileList = []
        dirIt = QDirIterator(basePath, filters, QDir.Files, QDirIterator.Subdirectories)

        while dirIt.hasNext():
            fileList.append(dirIt.next())

        return fileList

    def numpy_image_2_qimage(self, numpy_image):
        b, g, r = cv2.split(numpy_image)
        nimage = cv2.merge([r, g, b])
        height, width, channel = numpy_image.shape
        bytes_per_line = 3 * width
        image = QImage(nimage.data,
                       width,
                       height,
                       bytes_per_line,
                       QImage.Format_RGB888)

        return image

    def onTemplateProgressIncreased(self, newValue):
        self.progressBarTemplate.setValue(newValue)

    def onBaseImageProgressIncreased(self, newValue):
        self.progressBarImage.setValue(newValue)

    def onChangeStatusBarMessage(self, x, y, filepath):
        message = 'Coincidencia encontrada con el temlate : ' + filepath + \
                  ' en la posición: ' + str(x) + ' , ' + str(y)
        self.statusbar.showMessage(message)

    def setImageFocus(self, x, y):
        verMax = self.scrollArea.verticalScrollBar().maximum()
        horMax = self.scrollArea.horizontalScrollBar().maximum()
        self.scrollArea.verticalScrollBar().setValue(min(verMax, y))
        self.scrollArea.horizontalScrollBar().setValue(min(horMax, x))

    def saveTemplate(self, x, y, image, templateFileName):
        self.templateCount += 1
        height, width, _ = self.numpy_image.shape
        maxHeight = min (height, y + 128)
        maxWidth = min (width, x + 128)
        template = self.numpy_image[y:maxHeight, x:maxWidth]

        dirName = os.path.dirname(self.workingImagePath)
        #imageName = os.path.splitext(os.path.basename(self.workingImagePath))[0]
        imageName = getFileNameFromPath(self.workingImagePath)
        baseName = os.path.basename(dirName)
        templateName = baseName +'-'+imageName +'-'+ str(self.templateCount)+ '-' + templateFileName + '.png'
        savePath = 'C:/Users/home/New folder/' + templateName
        cv2.imwrite(savePath, template)

    def saveRemainingTemplateList(self, templateList):
        file = open('C:/Users/home/New folder/templateList.txt', 'w')
        for path in templateList:
            line = path.absoluteFilePath()
            file.write(line+'\n')
        file.close()

    templateProgressIncreased = pyqtSignal(int)
    templateFoundInImage = pyqtSignal(int, int)
    baseImageChanged = pyqtSignal(int)

if __name__ == "__main__":
    import sys
    app = wid.QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())
