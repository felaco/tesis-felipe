from keras.engine.training import Model
from keras.layers.core import Dense
from keras.models import model_from_config, model_from_json, Sequential
from keras.utils.np_utils import to_categorical
import numpy as np
import time
from common.utils import write_test_output


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def keras_model_deep_copy(keras_model):
    config = keras_model.get_config()
    if isinstance(keras_model, Sequential):
        new_model = Sequential.from_config(config)
    else:
        new_model = model_from_config(config)
    shuffle_weights(new_model)
    loss = keras_model.loss
    metrics = keras_model.metrics
    optimizer = keras_model.optimizer
    new_model.compile(optimizer, loss, metrics)
    return new_model


def _get_classes_index(y_true):
    """
    Create a list of the indexes of each class
    :param y_true: expected output of the model
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    class_indexes = []
    unique = np.unique(y_true)
    for val in unique:
        ind = y_true == val
        class_indexes.append(ind)

    return class_indexes


def _split_classes(x, y_true):
    """
    split the input in a list, which contains only members of a class on
    each index
    :param x: input of a model
    :param y_true: expected output, the labels
    :return: a list of samples of each class
    """
    indexes = _get_classes_index(y_true)
    class_samples = []
    for ind in indexes:
        sample = x[ind]
        class_samples.append(sample)

    return class_samples

def is_empty(list):
    if len(list) == 0:
        return True
    return False

class Bagging:
    def __init__(self,
                 estimator_func=None,
                 n_estimators=11,
                 max_samples=1.0,
                 bootstrap = True):

        self.estimators = []
        self.n_estimators = n_estimators
        if estimator_func is not None:
            for i in range(n_estimators):
                model = estimator_func()
                shuffle_weights(model)
                self.estimators.append(model)

        assert isinstance(max_samples, float)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if bootstrap is False and max_samples > 1.0:
            raise ValueError("max samples cannot be greater than 1 if bootstrap is false")

        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_classes = 0

    def set_estimator(self, base_estimator, n_classes=2):
        if isinstance(base_estimator, list):
            for estimator in base_estimator:
                assert isinstance(estimator, Model)

            self.estimators = base_estimator
            self.n_estimators = len(base_estimator)
            self.n_classes = n_classes
            return

        self.n_classes = n_classes
        for i in range(self.n_estimators):
            self.estimators.append(base_estimator())

    def fit (self,
             x,
             y,
             epochs_per_model=40,
             batch_size=64,
             callbacks=None,
             class_weight=None,
             sample_weight=None,
             validation_data=None):

        if  is_empty(self.estimators):
            raise RuntimeError('The Bagging estimators has not been configured,'
                               ' please use Bagging.configure()')

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        model_inputs = _split_classes(x, y)
        history_list = []
        i= 1

        t_init = time.time()
        for estimator in self.estimators:
            print('Estimador : {}'.format(i))
            i += 1
            xe, y = self._get_bagging_samples(model_inputs)
            ye = to_categorical(y)
            history = estimator.fit(xe, ye,
                                    batch_size=batch_size,
                                    epochs=epochs_per_model,
                                    callbacks=callbacks,
                                    class_weight=class_weight,
                                    sample_weight=sample_weight,
                                    validation_data=validation_data,
                                    verbose = 2)

            history_list.append(history)

        t_end = time.time()
        print('time: {} seg'.format(t_end - t_init))
        return history_list

    def _get_bagging_samples(self, splitted_samples):
        class_samples = self._get_bagging_samples_per_class(splitted_samples)
        # get the training samples of the first class
        # model input is a numpy array
        model_input = class_samples[0]
        model_expected_output= np.zeros(len(model_input))
        i=1
        # merges the samples of all classes in a single numpy array
        while i < len(class_samples):
            sample = class_samples[i]
            model_input = np.append(model_input, sample, axis=0)

            output = np.full(len(sample), fill_value=i, dtype=int)
            model_expected_output = np.append(model_expected_output, output)
            i += 1

        # the actual shuffling
        shuffle_index = np.arange(len(model_input))
        np.random.shuffle(shuffle_index)

        self.n_classes = len(class_samples)

        return model_input[shuffle_index], model_expected_output[shuffle_index]

    def _get_bagging_samples_per_class(self, splitted_samples):
        class_samples = []
        for sample in splitted_samples:
            size = len(sample)
            choices = np.random.choice(size,
                             size= int(size * self.max_samples),
                             replace=True)
            class_samples.append(sample[choices])

        return class_samples

    def predict_on_batch(self, x, y=None):
        if self.n_classes == 0:
            raise RuntimeError('The model has not been fitted')

        predict_shape = (len(x), self.n_classes)
        proba = np.zeros(predict_shape)

        name_count = 1
        for model in self.estimators:
            res = res = model.predict(x, verbose=0)
            res = np.argmax(res, 1)

            for i in range(len(x)):
                # stores the votes of each predictor
                proba[i, res[i]] += 1

            if y is not None:
                write_test_output(y, res, 'resultados{}'.format(name_count))
            name_count += 1

        return self._mayority_vote(proba)

    def predict(self, x):
        proba = np.zeros(self.n_classes)
        for model in self.estimators:
            res = model.predict(x, verbose=0)
            res = np.argmax(res, 1)[0]
            proba[res] += 1

        return self._mayority_vote(proba)

    def _mayority_vote(self, prediction_matrix):
        matrix_shape = prediction_matrix.shape
        # the prediction of only 1 sample
        if len(matrix_shape) == 1:
            return np.argmax(prediction_matrix)
        else:
            return np.argmax(prediction_matrix, axis=1)


if __name__ == '__main__':
    from mitosCalsification import loadDataset as ld
    from common.Params import Params as P

    train = ld.dataset(P().saveCutCandidatesDir + 'candidates.tar', P().saveMitosisPreProcessed)
    xe, ye = train.get_training_sample(selection=False)
    del train
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(2,)))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mse')
    bag = Bagging(model)
    bag.fit(xe, ye)
