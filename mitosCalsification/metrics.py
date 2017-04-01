import keras.backend as K

def mitos_fscore(y_true, y_pred):
    true_output = K.argmax(y_true)
    pred_output = K.argmax(y_pred)
    index_true_class = K.equal(true_output, 0)
    inverted_true_class = K.equal(index_true_class, 0)
    index_pred_class = K.equal(pred_output, 0)

    false_positive = K.sum(inverted_true_class * index_pred_class)
    true_positives = K.sum(K.equal(true_output, pred_output)* index_true_class)
    false_negatives = K.sum(index_true_class) - true_positives

    precision = true_positives / (true_positives + false_positive)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall)/(precision + recall)
    #print(K.eval(fscore))



if __name__ == '__main__':
    m1 = [[1,0],[1,0],[0,1], [1,0], [0,1], [0,1], [1,0],[0,1],[1,0],[1,0],[1,0]]
    m2 = [[1,0], [1,0], [1,0], [0,1], [0,1], [0,1], [1,0], [1,0],[1,0],[0,1],[0,1]]

    mitos_fscore(m2, m1)
    i=0
