import os

from keras import backend as K, optimizers
from keras.applications import vgg16
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.losses import binary_crossentropy
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy, binary_accuracy
import numpy as np
import sys

from sklearn.metrics.ranking import roc_curve

from common.Params import Params as P
from common.utils import listFiles
from mitosCalsification import metrics, loadDataset as ld
from mitosCalsification.End_training_callback import End_training_callback
from common.utils import getInputDim, write_test_output
from mitosCalsification.plot import print_plots, dump_metrics_2_file, plot_roc
from models.SimpleModel import create_simple_model, create_fel_res
from models.SqueezeNet import create_squeeze_net
from models.vgg16 import create_vgg16
from models.Bagging import Bagging

from sklearn.ensemble.bagging import BaggingClassifier

import time




def save_model(model, name):
    json_string = model.to_json()
    file = open('./saved_models/' + name + '.json','w')
    file.write(json_string)
    file.close()
    model.save_weights('./saved_models/' + name + '_weights.h5')


def load_model(name):
    file = open('./saved_models/' + name + '.json')
    json_string = file.read()
    file.close()
    model = model_from_json(json_string)
    model.load_weights('./saved_models/' + name + '_weights.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.mitos_fscore])
    return model


def save_bagging_model(bagging):
    i = 1
    for estimator in bagging.estimators:
        model_base_name = 'model' + str(i)
        save_model(estimator, model_base_name)
        i += 1


def load_bagging_model():
    filter = ['model*.json']
    info_list = listFiles('./saved_models/', filter)
    if len(info_list) == 0:
        raise FileNotFoundError('There is no model saved')

    estimators = []
    for file_info in info_list:
        model = load_model(file_info.baseName())
        estimators.append(model)

    bag = Bagging()
    bag.set_estimator(estimators)
    return bag


def load_test_data():
    if sys.platform == 'win32':
        cand_path = 'C:/Users/felipe/mitos dataset/eval/no-mitosis/candidates.tar'
        mit_path = 'C:/Users/felipe/mitos dataset/eval/mitosis/'
    else:
        cand_path = '/home/facosta/dataset/test/no-mitosis/candidates.tar'
        mit_path = '/home/facosta/dataset/test/mitosis/'

    # test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    test = ld.dataset(cand_path, mit_path)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    yt_cat = np_utils.to_categorical(yt)

    return xt, yt

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


def _do_train(model, train_data, val_data, epochs, batch_size):
    # generator = ImageDataGenerator(rotation_range=40,
    #                                width_shift_range=0.3,
    #                                height_shift_range=0.3,
    #                                shear_range=0.1,
    #                                zoom_range=0.2,
    #                                fill_mode='wrap',
    #                                horizontal_flip=True,
    #                                vertical_flip=True)
    generator = ImageDataGenerator()
    xe = train_data[0]
    # ye = np_utils.to_categorical(train_data[1])
    ye = train_data[1]
    xv = val_data[0]
    yv = val_data[1]
    # yv = np_utils.to_categorical(val_data[1])
    xt, yt = load_test_data()
    class_weight = _get_class_weights(ye)

    train_history_list =[]
    val_history_list = []
    test_history_list = []
    test_res = 0

    for e in range(epochs):
        print('Epoch: {}/{}'.format(e + 1, epochs))
        batches = 0
        start_time = time.time()
        history_list = []
        for x_batch, y_batch in generator.flow(xe, ye, batch_size):
            history = model.train_on_batch(x_batch,
                                           y_batch,
                                           class_weight=class_weight)
            history_list.append(history)
            batches += 1
            if batches >= int(len(xe) / batch_size):
                break

        history = np.asarray(history_list).mean(axis=0)
        train_history_list.append(history)
        history_names = model.metrics_names

        train_val = model.predict(xv, batch_size)
        train_val = np.amax(train_val, axis=1)
        val_loss = binary_crossentropy(K.variable(yv), K.variable(train_val))
        val_loss = K.eval(K.mean(val_loss))
        # train_val = np.argmax(train_val, axis=1)
        train_val = np.round(train_val, decimals=0).astype(int)
        val_fscore = metrics.fscore(val_data[1], train_val)
        val_history_list.append((val_loss, val_fscore))

        fscore,_,test_res,_ = _do_test(model, xt, yt)
        test_history_list.append(fscore)

        end_time = time.time()
        print('time: {:.1f}'.format(end_time - start_time), end=' - ')
        for name, value in zip(history_names, history):
            print(name +': {:.4f}'.format(value), end=' ')

        print('val_loss: {:.4f} val_mitos_fscore: {:.4f}'. format(val_loss, val_fscore), end=' ')
        print('test_fscore: {:.4f}'.format(fscore), flush=True)

    metrics.print_conf_matrix(yt, test_res)
    return np.transpose(train_history_list), np.transpose(val_history_list), test_history_list


def train_model(ratio, use_all):
    selection = True
    if use_all:
        selection = False
    elif ratio <= 0:
        raise ValueError('ratio cannot be neither negative nor 0')
    train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
    xe, ye = train.get_training_sample(ratio=ratio, selection=selection)

    model = create_fel_res()
    # model = create_simple_model()
    # model = create_squeeze_net()
    # bagging_classifier = Bagging(estimator_func=create_simple_model,
    #                              n_estimators=31,
    #                              max_samples=0.1,
    #                              bootstrap=True)

    class_weight = _get_class_weights(ye)
    epochs = P().model_epoch
    epochs = 40
    batch_size = 128

    test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    xval, yval = test.get_training_sample(shuffle=False, selection=False)
    yval_cat = np_utils.to_categorical(yval)

    del train, test # saves around 500 mb of ram. Wow!
    end_callback = End_training_callback()
    # model.fit(xe, target, epochs=epochs, verbose=2,
    #           class_weight=class_weight,
    #           validation_data=(xval, yval_cat),
    #           batch_size=128,
    #           callbacks=[end_callback])

    train_metric, val_metric, test_metrics =_do_train(model,(xe, ye), (xval, yval), epochs, 128)

    if sys.platform == 'win32':
        print_plots(model.metrics_names, train_metric, val_metric, test_metrics)
    else:
        dump_metrics_2_file(train_metric, val_metric, test_metrics)

    # bagging_classifier.fit(xe, ye, epochs,
    #                        batch_size=batch_size,
    #                        callbacks=[end_callback],
    #                        class_weight=class_weight,
    #                        validation_data=(xval, yval_cat))
    # save_bagging_model(bagging_classifier)
    save_model(model, 'model1')


def _do_test(model, xt, yt):
    if isinstance(model, Bagging):
        yt2 = np.argmax(yt, axis=1)
        res_rounded = model.predict_on_batch(xt, yt2)
        res = res_rounded
    else:
        res = model.predict(xt)
        # res = np.argmax(res, axis=1)
        res_rounded = np.round(res, decimals=0).astype(int)

    cat_res = np_utils.to_categorical(res_rounded)
    fscore = metrics.fscore(yt, res_rounded)
    prec = K.eval(K.mean(binary_accuracy(K.variable(yt),
                                              K.variable(res_rounded))))

    return fscore, prec, res_rounded, res


def test_model():
    import json
    from common.utils import Mitos_test_evaluator

    bag = load_model('model1')
    xt, yt = load_test_data()

    fscore, prec, res_round, res = _do_test(bag, xt, yt)
    metrics.print_conf_matrix(yt, res_round)
    print('fscore: {}'.format(fscore))
    print('precision: {}'.format(prec))
    write_test_output(yt, res_round)
    plot_roc(yt, res)


    # test_json_path = P().candidatesTestJsonPath
    # with open(test_json_path) as file:
    #     json_string = file.read()
    #     cand_dict = json.loads(json_string)
    #
    # evaluator = Mitos_test_evaluator(cand_dict)
    # for candidate in evaluator:
    #     res = model.predict(candidate, verbose=0)
    #     evaluator.add_prediction(np.argmax(res, 1)[0])
    #
    # evaluator.evaluate()
    # evaluator.print_conf_matrix()
    # evaluator.print_res_to_img()


if __name__ == '__main__':
    test = ld.dataset(P().saveTestCandidates + 'candidates.tar', P().saveTestMitos)
    xt, yt = test.get_training_sample(shuffle=False, selection=False)
    yt_cat = np_utils.to_categorical(yt)

    model = load_model('model3')
    res = model.predict_classes(xt)

    cat_res = np_utils.to_categorical(res)
    fscore = K.eval(metrics.mitos_fscore(K.variable(yt_cat),
                                         K.variable(cat_res)))

    metrics.print_conf_matrix(yt, res)
    print('fscore: {}'.format(fscore))
    write_test_output(yt, res)


