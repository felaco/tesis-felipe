from keras.engine.training import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.layers import concatenate
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import add
from keras.models import Sequential
from keras.layers import merge, add

from common.utils import getInputDim
from mitosCalsification import metrics
import keras.backend as K


def create_simple_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=getInputDim(),activation='relu', padding='same'))
    # model.add((Conv2D(32, (3,3), activation='relu', padding='same')))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))


    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=[metrics.mitos_fscore,'binary_accuracy'])

    return model

def create_highway(prev_layer, num_neurons, layerid):
    if K.image_data_format() == 'channels_last':
        merge_channel = 3
    else:
        merge_channel = 1

    x = Conv2D(num_neurons, (3,3), padding='same')(prev_layer)
    x = BatchNormalization(axis=merge_channel)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_neurons, (3,3), padding='same')(x)
    x = BatchNormalization(axis=merge_channel)(x)
    x = Activation('relu')(x)
    x = add([x, prev_layer])
    return x


def create_fel_res():
    inp = Input(getInputDim())
    x = Conv2D(32, (3,3), activation='relu')(inp)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3,3), strides=2)(x)
    x = create_highway(x, 64, 1)
    x = create_highway(x, 64, 1)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = create_highway(x, 64, 2)
    x = MaxPooling2D((3,3), strides=2)(x)
    x = Conv2D(32, (1,1), activation='relu')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Activation('softmax')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=[metrics.mitos_fscore, 'binary_accuracy'])

    return model


if __name__ == '__main__':
    model = create_fel_res()
    model.summary()