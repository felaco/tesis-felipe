import os

from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
import numpy as np

from common.Params import Params as P
from mitosCalsification import metrics, loadDataset as ld


def write_test_output(true_output, pred_output):
    i = 0
    file = open('resulados.txt', 'w')
    file.write('true\tpred\n')
    while i < len(true_output):
        file.write('{}\t\t{}\n'.format(true_output[i], pred_output[i]))
        i+=1

def save_model(model):
    json_string = model.to_json()
    file = open('model.json','w')
    file.write(json_string)
    file.close()
    model.save_weights('weights.h5')

def load_model():
    file = open('model.json')
    json_string = file.read()
    file.close()
    model = model_from_json(json_string)
    model.load_weights('weights.h5')
    return model

def getInputDim():
    img_width = 63
    img_height = 63

    if K._image_data_format == 'channels_first':
        dim = (3, img_width, img_height)
    else:
        dim = (img_width, img_height, 3)

    return dim

def _get_class_weights(labels):
    import math
    weight_dict = {}
    total = len(labels)
    unique = np.unique(labels, return_counts=True)
    i = 0
    while i < len(unique):
        classes_count = unique[1]
        class_count = classes_count[i]
        weight = math.log(total / class_count, 1.7) # base = 1.7
        if weight < 1:
            weight = 1
        weight_dict[i] = weight
        i += 1

    return weight_dict

def createModel():
    model = Sequential()
    model.add(Convolution2D(32, (3,3), input_shape=getInputDim() , activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (1, 3), activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))
    #model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[metrics.mitos_fscore])
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

def train_model(ratio, use_all):
    selection = True
    if use_all:
        selection = False
    elif ratio <= 0:
        raise ValueError('ratio cannot be neither negative nor 0')
    train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
    xe, ye = train.get_training_sample(ratio=ratio, selection=selection)

    target = np_utils.to_categorical(ye)
    model = createModel()
    # res = fitAndEvaluate(model, xe, target,)
    # print(res)
    model.fit(xe, target, epochs=30, verbose=2, validation_split=0.1)
    save_model(model)


def test_model():
    test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    yt_cat = np_utils.to_categorical(yt)

    if os.path.isfile('model.json'):
        model = load_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=[metrics.mitos_fscore])
    else:
        raise FileNotFoundError('There is no model saved')

    #res = model.evaluate(xt, yt2, verbose=0)
    res = model.predict_classes(xt, verbose=0)
    cat_res = np_utils.to_categorical(res)
    fscore = K.eval(metrics.mitos_fscore(yt_cat, cat_res))
    print(fscore)
    write_test_output(yt, res)

if __name__ == '__main__':
    train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
    xe, ye = train.get_training_sample(selection=False)
    _get_class_weights(ye)
