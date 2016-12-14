import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from PyQt5.QtCore import QFile
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold

import csv
import loadDataset
import copyFalsePositives as fp
import time
import cv2



img_width = 63
img_height = 63

def createModel():
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(3, img_width, img_height), activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy','fmeasure'])
    '''
    model.fit_generator(
            train_generator,
            samples_per_epoch=1000,
            nb_epoch=30,
            validation_data=train_generator,
            nb_val_samples=600,
            verbose=2)
    '''
    #model.save('model2.h5')

    return model

def VGG_16(numClasses):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,63,63)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def fitAndEvaluate(model, trainInput, trainTarget, testInput, testTarget):
    model.fit(trainInput, trainTarget, nb_epoch=30, verbose=2, batch_size=32,
              validation_data=(testInput, testTarget))
    return model.evaluate(testInput, testTarget, verbose=0)

'''
validation_data_dir ='C:/Users/home/Desktop/mitos dataset eval/test'
train_data_dir = 'C:/Users/home/Desktop/mitos dataset/train no preprocesed'

train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='wrap')

train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            class_mode='categorical',
            batch_size=64)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            class_mode='categorical',
            batch_size=64)
'''

dataset = loadDataset.loadMitosDatasetTest()
x_test = dataset['xv']
y_test = dataset['yv']
y_test = np_utils.to_categorical(y_test)
x_train = dataset['xe']
y_train = dataset['ye']
y_train_cat = np_utils.to_categorical(y_train)
finalVal = dataset['finalVal']
finalValTarget = np.ones(len(finalVal))
finalValTarget = np_utils.to_categorical(finalValTarget)


seed = 7
np.random.seed(seed)

skf = StratifiedKFold(y_train, n_folds=10, shuffle=True, random_state=seed)
j=0
metricsList = []
metricsNames = 0

for i, (train, test) in enumerate(skf):
    xe = x_train[train]
    ye = y_train_cat[train]
    xv = x_train[test]
    yv = y_train_cat[test]
    model = None
    model = createModel()
    metrics = fitAndEvaluate(model, xe, ye, xv, yv)
    metrics = model.evaluate(finalVal, finalValTarget, verbose=0)
    metricsList.append(metrics)
    metricsNames = model.metrics_names
    print(metricsNames)
    print(metrics)
    break

''''

with open('resultados.txt', 'w') as mycsv:
    writer = csv.writer(mycsv)
    writer.writerow(metricsNames)
    for row in metricsList:
        writer.writerow(row)

'''

'''
savedModel = QFile('model.h5')
if savedModel.exists():
    model = load_model('model.h5')
else:
    model = createModel()
    x_train = dataset['xe']
    y_train = dataset['ye']
    y_train = np_utils.to_categorical(y_train)
    model.fit(x_train, y_train, nb_epoch=25, verbose=2, batch_size=32,
              validation_data=(x_test, y_test))
    #model.fit_generator(train_generator,
    #                    samples_per_epoch=640,
    #                    nb_epoch=30,
    #                    verbose=2,
    #                    validation_data=validation_generator,
    #                    nb_val_samples=320)
    model.save('model.h5')
'''

'''
y_test = np_utils.to_categorical(y_test)
savedModel = QFile('vgg16.h5')
if savedModel.exists():
    model = load_model('vgg16.h5')
else:
    model = VGG_16(2)
    x_train = dataset['xe']
    y_train = dataset['ye']
    y_train = np_utils.to_categorical(y_train)

    #model.fit(x_train, y_train, validation_data=(x_test, y_test),verbose=2,
    #          batch_size=128)

    model.fit_generator(train_generator, 2000, 25, verbose=2, validation_data=validation_generator,nb_val_samples=500)
    model.save('vgg16.h5')
'''

'''
validation = dataset['noMitosTest']
validationFileName = dataset['noMitosFile']

res = model.predict_classes(validation, verbose=0)
score = model.evaluate(x_test, y_test)
#TENER MUCHO CUIDADO PORQ BORRA ARCHIVOS!!!
#fp.copyFalsePositives(validationFileName, res) #por eso ta comentado
noZero = cv2.countNonZero(res)
print('\n%f\n'%(noZero/len(res),))
print (score)
'''

'''
noMitos, mitos, nameNoMitos, nameMitos = loadDataset.loadMitosDatasetTest()
resNoMitos = model.predict_classes(noMitos)
resMitos = model.predict_classes(mitos)
noZero = cv2.countNonZero(resNoMitos)
print('precicion no mitos: %f\n'%(noZero / len(resNoMitos),))
noZero = cv2.countNonZero(resMitos)
print('precicion mitos: %f\n'%(1- (noZero/len(resMitos))),)
nc = np.ones(len(noMitos))
mc = np.zeros(len(mitos))
yv = np.append(nc, mc, axis=0)
xv = np.append(noMitos, mitos, axis=0)
#yv = np_utils.to_categorical(yv)

score = model.evaluate(xv,yv)
names = model.metrics_names
i = 0
'''
