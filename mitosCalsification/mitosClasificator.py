import os

from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

from common.Params import Params as P
from mitosCalsification import metrics, loadDataset as ld


def write_test_output(true_output, pred_output):
    i = 0
    file = open('resul.txt', 'w')
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


train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
xe, ye = train.get_training_sample()
test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
xt, yt = test.get_training_sample(shuffle=False, selection=False)
yt2 = np_utils.to_categorical(yt)

target = np_utils.to_categorical(ye)

if os.path.isfile('model.json'):
    model = load_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[metrics.mitos_fscore])
else:
    model = createModel()
    # res = fitAndEvaluate(model, xe, target,)
    # print(res)
    model.fit(xe, target, epochs=10, verbose=2, validation_split=0.1)
    save_model(model)



res = model.evaluate(xt, yt2, verbose=0)
res2 = model.predict_classes(xt, verbose=0)
print(res)
write_test_output(yt, res2)
