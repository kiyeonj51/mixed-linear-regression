from mlr.cores.data import *
from mlr.cores.loss import *
from mlr.utils.initialize import *
from mlr.cores.base import (EM_HD, EMP, RR_HD, EMRR, RRONE_HD, ARR_HD)

# Example setting
params = {
    "data_name": DataMLR,
    "init_method": init_rand_normal,
    "reg": 1.,
    "n": 1000,
    "d": 10000,
    "s": 10,
    "max_iter": 2000,
    "p1": .5,
    "std": 0,
    "num_simulation": 10
}
funcs = (func_mlr,  None, None)

mlr_algorithms = [
    ("ARR_HD", ARR_HD(funcs=funcs, max_iter=params['max_iter'], info="Adaptive RR", reg=params['reg'])),
]

for i in range(params['num_simulation']):
    print(f'\n{i}-th simulation')
    # generate data
    dataset = params['data_name']
    data = dataset.generate_data(params)

    # optimize
    results = []
    for name, algorithm in mlr_algorithms:
        res = algorithm.solve(data=data, init_method=params['init_method'])
        results.append(res)
