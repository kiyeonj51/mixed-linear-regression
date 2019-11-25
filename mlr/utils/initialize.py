import numpy as np
import numpy.linalg as la

def init_zero(data, p1):
    n, d = data['n'], data['d']
    beta = np.zeros(2 * d)
    variable = {'beta': beta}
    return variable


def init_ones(data, p1):
    n, d = data['n'], data['d']
    beta = np.ones(2 * d)
    variable = {'beta': beta}
    return variable


def init_random(data, p1):
    n = data['n']
    S = np.random.randint(n, size=int(n * p1))
    X1 = data['X'][S, :]
    Y1 = data['Y'][S]
    X2 = data['X'][~S, :]
    Y2 = data['Y'][~S]
    beta1 = la.solve(X1.T.dot(X1), X1.T.dot(Y1))
    beta2 = la.solve(X2.T.dot(X2), X2.T.dot(Y2))
    beta = np.concatenate((beta1, beta2))
    variable = {'beta': beta}
    return variable


def init_rand_normal(data, p1):
    d = data['d']
    beta1 = np.random.normal(0, 1, d)
    beta2 = np.random.normal(0, 1, d)
    # beta1 /= np.linalg.norm(beta1)
    # beta2 /= np.linalg.norm(beta2)
    beta = np.concatenate((beta1, beta2))
    variable = {'beta': beta}
    return variable


def init_rand_uniform(data, p1):
    d = data['d']
    beta1 = np.random.uniform(-1, 1, d)
    beta2 = np.random.uniform(-1, 1, d)
    # beta1 = np.random.rand(d)
    # beta2 = np.random.rand(d)
    # beta1 /= np.linalg.norm(beta1)
    # beta2 /= np.linalg.norm(beta2)
    beta = np.concatenate((beta1, beta2))
    variable = {'beta': beta}
    return variable
