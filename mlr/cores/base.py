from abc import ABC, abstractmethod
from time import *
from mlr.utils.initialize import *
import numpy.linalg as la
from mlr.utils.operator import softth


# Low dim
class OPT(ABC):
    def __init__(self, funcs, max_iter=100, reg=0, info=None):
        self.funcs = funcs
        self.max_iter = max_iter
        self.reg = reg
        self.info = info

    def solve(self, data, init_method=init_rand_normal):
        max_iter = self.max_iter
        info = self.info
        reg = self.reg
        func, grad, hess = self.funcs
        train = data['train']
        test = data['test']
        d = data['d']
        p1 = data['p1']
        beta_gt_12 = data['beta_gt']
        beta_gt_21 = np.concatenate((beta_gt_12[d:], beta_gt_12[:d]))

        # initialization
        betas, tr_errors, tr_regs, te_errors, elapsed_times = [], [], [], [], []
        variable = init_method(train, p1)

        # optimization
        tr_error = tr_reg = te_error = beta_error = np.inf
        variable['beta_error'] = beta_error
        for iteration in range(max_iter):
            te_z = self.update_sample(variable, test, p1, iteration)
            start = time()
            tr_z = self.update_sample(variable, train, p1, iteration)
            variable = self.update(variable, train, tr_z)
            end = time()
            beta = variable['beta']
            # beta1, beta2 = beta[:d], beta[d:]
            # beta1 /= np.linalg.norm(beta1)
            # beta2 /= np.linalg.norm(beta2)
            # beta = np.concatenate((beta1, beta2))
            tr_error, tr_reg = func(beta, train, tr_z, reg)
            te_error = func(beta, test, te_z, reg)[0]
            tr_errors.append(tr_error)
            tr_regs.append(tr_reg)
            te_errors.append(te_error)
            betas.append(beta)
            elapsed_times.append(end - start)
            beta_gt = beta_gt_12 if (la.norm(beta - beta_gt_12) < la.norm(beta - beta_gt_21)) else beta_gt_21
            beta_error = max(la.norm((beta_gt - beta)[:d]), la.norm((beta_gt - beta)[d:]))
            variable['beta_error'] = beta_error
        print(
            f'{info}|tr_error : {tr_error:3.6f}, reg : {tr_reg:.7e}, te_error : {te_error:3.6f}, beta_error : {beta_error:3.6f}')

    def update(self, variable, data, mlr_subset):
        X = data['X']
        Y = data['Y']
        d = data['d']
        X1, Y1 = X[mlr_subset, :], Y[mlr_subset]
        X2, Y2 = X[~mlr_subset, :], Y[~mlr_subset]
        beta = variable['beta']
        beta1, beta2 = beta[:d], beta[d:]
        # if np.sum(mlr_subset) != 0:
            # print(np.sum(mlr_subset))
        beta1 = la.solve(X1.T.dot(X1), X1.T.dot(Y1))
        # if np.sum(~mlr_subset) != 0:
            # print(np.sum(mlr_subset))
        beta2 = la.solve(X2.T.dot(X2), X2.T.dot(Y2))
        beta = np.concatenate((beta1, beta2))
        variable['beta'] = beta
        return variable

    @abstractmethod
    def update_sample(self, variable, data, p1, iter):
        raise NotImplementedError


class EM(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        S = (abs(X.dot(beta1) - Y) < abs(X.dot(beta2) - Y))
        return S


class RR(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        samples = np.argsort(dist)[:int(n * p1)]
        S = np.asarray([False] * n)
        S[samples] = True
        return S


class EMRR(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta_error = variable['beta_error']
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        if beta_error < 0.1:
            beta1, beta2 = beta[:d], beta[d:]
            S = (abs(X.dot(beta1) - Y) < abs(X.dot(beta2) - Y))
        else:
            dist = abs(X.dot(beta1) - Y)
            samples = np.argsort(dist)[:int(n * p1)]
            S = np.asarray([False] * n)
            S[samples] = True
        return S


class RRONE(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']

        # num_sample = max(n-(params['iteration']+1), int(n * p1))
        # beta1, beta2 = beta[:d], beta[d:]
        # dist = abs(X.dot(beta1)-Y)
        # samples = np.argsort(dist)[:num_sample]
        # S = np.asarray([False] * n)
        # S[samples] = True
        num_sample = max(n - (iter+ 1), int(n * p1))
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        samples = np.argsort(dist)[:num_sample]
        S = np.asarray([False] * n)
        S[samples] = True

        # if (n - (params['iteration'] + 1)) > int(n * p1):
        #     num_sample = max(n - (params['iteration'] + 1), int(n * p1))
        #     samples = np.argsort(dist)[:num_sample]
        #     S = np.asarray([False] * n)
        #     S[samples] = True
        # else:
        #     S = (abs(X.dot(beta1) - Y) < abs(X.dot(beta2) - Y))
        return S


class ARR(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        S = dist < (20./ np.sqrt(iter+1))
        return S


class EMP(OPT):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        ps = abs(X.dot(beta2) - Y) / (abs(X.dot(beta1) - Y) + abs(X.dot(beta2) - Y))
        S = (np.asarray([np.random.binomial(1, p) for p in ps]) == 1)
        return S


# High dim
class OPT_HD(ABC):
    def __init__(self, funcs, max_iter=100, reg=0, info=None):
        self.funcs = funcs
        self.max_iter = max_iter
        self.reg = reg
        self.info = info

    def solve(self, data, init_method=init_rand_normal):
        max_iter = self.max_iter
        info = self.info
        reg = self.reg
        func, grad, hess = self.funcs
        train = data['train']
        test = data['test']
        d = data['d']
        p1 = data['p1']
        beta_gt_12 = data['beta_gt']
        beta_gt_21 = np.concatenate((beta_gt_12[d:], beta_gt_12[:d]))

        # initialization
        betas, tr_errors, tr_regs, te_errors, elapsed_times = [], [], [], [], []
        variable = init_method(train, p1)

        # optimization
        tr_error = tr_reg = te_error = beta_error = np.inf
        variable['beta_error'] = beta_error
        for iteration in range(max_iter):
            te_z = self.update_sample(variable, test, p1, iteration)
            start = time()
            tr_z = self.update_sample(variable, train, p1, iteration)
            variable = self.update(variable, train, tr_z)
            end = time()
            beta = variable['beta']
            beta1, beta2 = beta[:d], beta[d:]
            beta1 /= np.linalg.norm(beta1)
            beta2 /= np.linalg.norm(beta2)
            beta = np.concatenate((beta1, beta2))
            tr_error, tr_reg = func(beta, train, tr_z, reg)
            te_error = func(beta, test, te_z, reg)[0]
            tr_errors.append(tr_error)
            tr_regs.append(tr_reg)
            te_errors.append(te_error)
            betas.append(beta)
            elapsed_times.append(end - start)
            beta_gt = beta_gt_12 if (la.norm(beta - beta_gt_12) < la.norm(beta - beta_gt_21)) else beta_gt_21
            beta_error = max(la.norm((beta_gt - beta)[:d]), la.norm((beta_gt - beta)[d:]))
            variable['beta_error'] = beta_error
            if beta_error < 0.02:
                print("Done!")
                break
            if iteration % int(max_iter/10) == 0:
                # print(set(np.sort(np.where(beta != 0)[0])).intersection(set(np.sort(np.where(beta_gt != 0)[0]))))
                print(f'{info}|tr_error : {tr_error:3.6f}, reg : {tr_reg:.7e}, te_error : {te_error:3.6f}, beta_error : {beta_error:3.6f}')

    def update(self, variable, data, mlr_subset):
        X = data['X']
        Y = data['Y']
        d = data['d']
        s = data['s']
        reg = self.reg
        beta = variable['beta']
        beta1 = beta[:d]
        beta2 = beta[d:]
        X1, Y1 = X[mlr_subset, :], Y[mlr_subset]
        X2, Y2 = X[~mlr_subset, :], Y[~mlr_subset]
        n1 = Y1.shape[0]
        n2 = Y2.shape[0]
        if n1 > 0:
            for ii in range(1):
                beta1 = beta1 - reg/(1+np.sqrt(ii))*(1. / n1) * X1.T.dot(X1.dot(beta1) - Y1)
                beta1_old = beta1
                idx = np.argsort(np.abs(beta1_old.flatten()))[-s:]
                beta1 = np.zeros(d)
                beta1[idx] = beta1_old[idx]
                # beta1 = softth(beta1_old, reg)
                # beta1 /= np.linalg.norm(beta1)
        if n2 > 0:
            for ii in range(1):
                beta2 = beta2 - reg/(1+np.sqrt(ii))*(1. / n2) * X2.T.dot(X2.dot(beta2) - Y2)
                beta2_old = beta2
                idx = np.argsort(np.abs(beta2_old.flatten()))[-s:]
                beta2 = np.zeros(d)
                beta2[idx] = beta2_old[idx]
                # beta2 = softth(beta2_old, reg)
                # beta2 /= np.linalg.norm(beta2)
        beta = np.concatenate((beta1, beta2))
        variable['beta'] = beta
        return variable

    @abstractmethod
    def update_sample(self, variable, data, p1, iter):
        raise NotImplementedError


class EM_HD(OPT_HD):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        S = (abs(X.dot(beta1) - Y) < abs(X.dot(beta2) - Y))
        return S


class RR_HD(OPT_HD):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        samples = np.argsort(dist)[:int(n * p1)]
        S = np.asarray([False] * n)
        S[samples] = True
        return S


class RRONE_HD(OPT_HD):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']

        # num_sample = max(n-(params['iteration']+1), int(n * p1))
        # beta1, beta2 = beta[:d], beta[d:]
        # dist = abs(X.dot(beta1)-Y)
        # samples = np.argsort(dist)[:num_sample]
        # S = np.asarray([False] * n)
        # S[samples] = True
        num_sample = max(n - (iter+ 1), int(n * p1))
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        samples = np.argsort(dist)[:num_sample]
        S = np.asarray([False] * n)
        S[samples] = True

        # if (n - (params['iteration'] + 1)) > int(n * p1):
        #     num_sample = max(n - (params['iteration'] + 1), int(n * p1))
        #     samples = np.argsort(dist)[:num_sample]
        #     S = np.asarray([False] * n)
        #     S[samples] = True
        # else:
        #     S = (abs(X.dot(beta1) - Y) < abs(X.dot(beta2) - Y))
        return S


class EMP_HD(OPT_HD):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        ps = abs(X.dot(beta2) - Y) / (abs(X.dot(beta1) - Y) + abs(X.dot(beta2) - Y))
        S = (np.asarray([np.random.binomial(1, p) for p in ps]) == 1)
        return S


class ARR_HD(OPT_HD):
    def update_sample(self, variable, data, p1, iter):
        beta = variable['beta']
        d = data['d']
        n = data['n']
        X = data['X']
        Y = data['Y']
        beta1, beta2 = beta[:d], beta[d:]
        dist = abs(X.dot(beta1) - Y)
        S = dist < (10./ np.sqrt(iter+1))
        return S