import numpy.linalg as la
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def func_mlr(beta, data, mlr_subset, reg=0):
    X = data['X']
    Y = data['Y']
    X1 = data['X'][mlr_subset, :]
    Y1 = data['Y'][mlr_subset]
    X2 = data['X'][~mlr_subset, :]
    Y2 = data['Y'][~mlr_subset]
    n = Y.shape[0]
    d = data['d']
    beta1 = beta[:d]
    beta2 = beta[d:]
    loss = (1. / n) * (la.norm(X1.dot(beta1)-Y1)**2 + la.norm(X2.dot(beta2)-Y2)**2)
    regularizer = reg/2. * (la.norm(beta1)**2 + la.norm(beta2)**2)
    return loss, regularizer