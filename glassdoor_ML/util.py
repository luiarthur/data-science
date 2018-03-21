import numpy as np

def confusion(y_pred, y_truth, numClasses):
    N = len(y_pred)
    assert N == len(y_truth)
    C = np.zeros( (numClasses, numClasses) )

    for i in range(N):
        C[y_pred[i], y_truth[i]] += 1

    return C

