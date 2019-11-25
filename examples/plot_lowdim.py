from mlr.cores.data import *
from mlr.cores.loss import *
from mlr.utils.initialize import *
from mlr.cores.base import (EM, EMP, RR, EMRR, RRONE, ARR)

# Example setting
params = {
    "data_name": DataMLR,
    "init_method": init_rand_normal,
    "reg": 0,
    "n": 1000,
    "d": 50,
    "s": 50,
    "max_iter": 100,
    "p1": .7,
    "std": 0,
    "num_simulation": 10
}
funcs = (func_mlr,  None, None)

mlr_algorithms = [
    ("EM", EM(funcs=funcs, max_iter=params['max_iter'], info="EM")),
    ("EMP", EMP(funcs=funcs, max_iter=params['max_iter'], info="EMP")),
    ("RR", RR(funcs=funcs, max_iter=params['max_iter'], info="RR")),
    ("EMRR", EMRR(funcs=funcs,max_iter=params['max_iter'], info="EMRR")),
    ("RRONE", RRONE(funcs=funcs, max_iter=params['max_iter'], info="RRONE")),
    # ("ARR", ARR(funcs=funcs, max_iter=params['max_iter'], info="ARR"))
]

for i in range(params['num_simulation']):
    print(f'\n{i}-th simulation')
    # generate data
    dataset = params['data_name']
    data = dataset.generate_data(params)

    # optimize
    results = []
    for name, algorithm in mlr_algorithms:
        res = algorithm.solve(data=data,init_method=params['init_method'])
        results.append(res)
