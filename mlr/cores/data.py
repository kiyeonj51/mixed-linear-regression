import os
import numpy as np
from keras.datasets import mnist


class DataMLR:
    @staticmethod
    def generate_data(params):
        n, d, s, p1, std = params['n'], params['d'], params['s'], params['p1'], params['std']

        num_tr = int(n*.7)
        num_te = n - num_tr
        # beta1_gt = np.random.uniform(-1, 1, d)
        beta1_gt = np.random.rand(d)*2-1
        beta1_gt[np.random.choice(d, d - s, replace=False)] = 0
        beta1_gt /= np.linalg.norm(beta1_gt)

        # beta2_gt = np.random.uniform(-1, 1, d)
        beta2_gt = np.random.rand(d) * 2 - 1
        beta2_gt[np.random.choice(d, d - s, replace=False)] = 0
        beta2_gt /= np.linalg.norm(beta2_gt)
        print('inner product of two vectors is',beta1_gt.dot(beta2_gt))
        beta_gt = np.concatenate((beta1_gt, beta2_gt))

        # generate data
        x = np.random.normal(0, 1, size=(n, d))
        y = np.append(x[:int(n * p1), :].dot(beta1_gt), x[int(n * p1):, :].dot(beta2_gt))
        y += np.random.normal(0, std, n)
        z = np.asarray([1] * int(n * p1) + [2] * (n - int(n * p1)))
        idx = list(range(n))
        np.random.shuffle(idx)
        x, y, z = x[idx, :], y[idx], z[idx]

        # divide train/test
        x_train, x_test = x[:int(num_tr),:], x[int(num_tr):,:]
        y_train, y_test = y[:int(num_tr)], y[int(num_tr):]
        z_train, z_test = z[:int(num_tr)], z[int(num_tr):]
        train = {
            'X': x_train, 'Y': y_train, 'Z':z_train, 'n': num_tr, 'd': d, 's': s
        }
        test = {
            'X': x_test, 'Y': y_test, 'Z': z_test, 'n': num_te, 'd': d, 's': s
        }
        data = {
            'train': train, 'test': test, 'beta_gt': beta_gt, 'd': d, 'p1': p1
        }
        return data
