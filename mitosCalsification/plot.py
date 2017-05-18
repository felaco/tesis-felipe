import numpy as np
import os

from sklearn.metrics import roc_curve
from keras.utils.np_utils import to_categorical


def print_plots(metrics_names, train_metrics, val_metrics=None, test_metrics=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x_axis = np.arange(1, len(train_metrics[0]) + 1, dtype=int)
    plt.plot(x_axis, train_metrics[0], label=metrics_names[0])

    i = 1
    style = ['--', '-.', ':']
    while i < len(train_metrics):
        plt.plot(x_axis, 1 - train_metrics[i],style[min(2, i)], label='error_'+metrics_names[i])
        i += 1

    plt.xlabel('iteración')
    plt.legend()
    plt.savefig('train.png')
    if val_metrics is not None:
        plt.figure()
        min_index = np.argmax(val_metrics[1])
        max_fscore = val_metrics[1][min_index]
        min_index += 1
        plt.plot(x_axis, val_metrics[0], label='val_' + metrics_names[0])
        plt.plot(x_axis, 1 - val_metrics[1],'--', label='val_error_' + metrics_names[1])
        print_text = '({}, {:.2f})'.format(min_index, 1 - max_fscore)
        plt.plot(min_index, 1 - max_fscore, 'ro')
        plt.text(min_index, 1 - max_fscore, print_text)
        plt.legend()
        plt.xlabel('iteración')
        plt.savefig('validation.png')

    if test_metrics is not None:
        plt.figure()
        plt.plot(x_axis, test_metrics, label='fscore')

        max_index = np.argmax(test_metrics)
        max_fscore = test_metrics[max_index]
        max_index += 1
        plt.plot(max_index, max_fscore, 'ro')
        print_text = '({}, {:.2f})'.format(max_index, max_fscore)
        plt.text(max_index, max_fscore, print_text)

        plt.legend()
        plt.xlabel('iteración')
        plt.savefig('test.png')

    # plt.show()

def dump_metrics_2_file(train_metrics, val_metrics=None, test_metrics=None):
    np.savetxt('train.csv', train_metrics, delimiter=',')
    if val_metrics is not None:
        np.savetxt('validation.csv', val_metrics, delimiter=',')

    if test_metrics is not None:
        np.savetxt('test.csv', test_metrics, delimiter=',')

def plot_from_file():
    train_metrics = np.loadtxt('train.csv', delimiter=',')
    metrics_names = ['loss', 'mitos_fscore', 'binary_accuracy']
    val_metrics = None
    test_metrics = None

    if os.path.isfile('validation.csv'):
        val_metrics = np.loadtxt('validation.csv', delimiter=',')

    if os.path.isfile('test.csv'):
        test_metrics = np.loadtxt('test.csv', delimiter=',')

    print_plots(metrics_names, train_metrics, val_metrics, test_metrics)

def plot_roc(y_true, y_pred):
    from matplotlib import pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label='Binary_roc', lw=1)

    y_pred = np.round(y_pred, decimals=0).astype(int)
    y_true_cat = to_categorical(y_true)
    y_pred_cat = to_categorical(y_pred)
    labels = ['Roc_mitosis', 'Roc_no_mitosis']
    style = []

    for i in range (2):
        fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_cat[:, i])
        plt.plot(fpr, tpr, label= labels[i], lw=1)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_from_file()
