import base64
import importlib
import json
import pickle
from functools import cache, lru_cache

import yaml

import example_problems.netlib_problems
import numpy as np

from collections import namedtuple

from bounds.bound_utils import *

@lru_cache(1)
def load_problem_from_lib(problem_name):
    return importlib.import_module("example_problems."+problem_name).problem

lib_problems = {
    "belgian_model": "three_clusters",
    "belgian_model_small": "three_month_three_clusters",
    "toy_1": "random_1",
    "toy_2": "random_2",
    "toy_3": "random_3",
    "toy_4": "random_4",
    "microgrid": "microgrid_small"
}

def _gen(y):
    return lambda: load_problem_from_lib(y)

def _gen2(x):
    return lambda: example_problems.netlib_problems.load_netlib_problem(x)

PROBLEMS = {
    x: _gen(y)
    for x, y in lib_problems.items()
} | {
    f"netlib_{x}": _gen2(x)
    for x in example_problems.netlib_problems.all_problems
}

if __name__ == '__main__':
    for entry in sorted(PROBLEMS.keys()):
        print(f"- {entry}")

def unpickle_bound(pickled):
    return pickle.loads(base64.b64decode(pickled))

def load_result_and_bounds(filename):
    data = json.load(open(filename))
    data["bounds"] = [{"timing": x["timing"], "bound": unpickle_bound(x["bound"])} for x in data["bounds"]]
    return data

def is_all_leq_or_close(a, b):
    return ((a <= b) | (np.isclose(a, b))).all()

ErrorMeasure = namedtuple("Error", ["abs", "rel", "availability"])
def compute_errors(problem, result, truth):
    space = np.linspace(*problem.range, truth.shape[0])
    bounds = [x["bound"](space) if x["bound"] is not None else Error()(space) for x in result["bounds"]]

    if len(bounds) == 0:
        bounds = [Error()(space)]

    if result["bound_type"] == "lb":
        bounds = np.max(np.nan_to_num(bounds, nan=-np.inf), axis=0)

        invalid_part = np.isnan(bounds) | np.isinf(bounds) | np.isnan(truth) | np.isinf(truth)
        if not is_all_leq_or_close(bounds[~invalid_part], truth[~invalid_part]):
            assert False
    else:
        bounds = np.min(np.nan_to_num(bounds, nan=np.inf), axis=0)

        invalid_part = np.isnan(bounds) | np.isinf(bounds) | np.isnan(truth) | np.isinf(truth)
        if not is_all_leq_or_close(truth[~invalid_part], bounds[~invalid_part]):
            assert False


    return compute_errors_from_values(bounds, truth)

def compute_errors_from_values(values, truth):
    min_v = truth.min()
    max_v = truth.max()

    error = truth - values

    isinf = np.isinf(error)
    availability = (error.shape[0] - isinf.sum()) / error.shape[0]

    error = error[~isinf]
    rmse = np.sqrt(np.average(error ** 2))
    return ErrorMeasure(rmse, rmse / (max_v - min_v), availability)