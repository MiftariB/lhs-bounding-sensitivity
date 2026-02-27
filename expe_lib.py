import base64
import importlib
import json
import pickle
from functools import lru_cache

import example_problems.netlib_problems
import example_problems.miplib_problems
import example_problems.facility_problems
import example_problems.unit_problems
import numpy as np

from collections import namedtuple

from bounds.bound_utils import *

@lru_cache(1)
def load_problem_from_lib(problem_name):
    return importlib.import_module("example_problems."+problem_name).problem

lib_problems = {
    #"belgian_model": "three_clusters",
    #"belgian_model_small": "three_month_three_clusters",
    "toy_1": "random_1",
    "toy_2": "random_2",
    "toy_3": "random_3",
    "toy_4": "random_4",
    "microgrid": "microgrid_small"
}

def pregen_problem(is_milp, problem):
    if not is_milp:
        problem.dual().positive()
        problem.positive()
    else:
        problem.positive()
        problem.linear().positive()
    return problem

def _gen(y):
    return lambda: pregen_problem(False, load_problem_from_lib(y))

def _gen_netlib(x):
    return lambda: pregen_problem(False, example_problems.netlib_problems.load_netlib_problem(x))

def _gen_miplib(x):
    return lambda: pregen_problem(True, example_problems.miplib_problems.load_miplib_problem(x))

def _gen_facility(x):
    return lambda: pregen_problem(True, example_problems.facility_problems.load_facility_problem(x))

def _gen_unit(x):
    return lambda: pregen_problem(True, example_problems.unit_problems.load_unit_problem(x))

PROBLEMS_NETLIB = {
    f"netlib_{x}": _gen_netlib(x)
    for x in example_problems.netlib_problems.all_problems
}

PROBLEMS_LP_OTHER = {
    x: _gen(y)
    for x, y in lib_problems.items()
}

PROBLEMS_LP = PROBLEMS_NETLIB | PROBLEMS_LP_OTHER

PROBLEMS_MIPLIB = {
    f"miplib_{x}": _gen_miplib(x)
    for x in example_problems.miplib_problems.all_problems
}

PROBLEMS_FACILITY = {
    f"facility_{x}": _gen_facility(x)
    for x in example_problems.facility_problems.all_problems
}

PROBLEMS_UNIT = {
    f"unit_{x}": _gen_unit(x)
    for x in example_problems.unit_problems.all_problems
}

PROBLEMS_MIP = PROBLEMS_MIPLIB | PROBLEMS_FACILITY | PROBLEMS_UNIT

PROBLEMS =  PROBLEMS_LP | PROBLEMS_MIP

if __name__ == '__main__':
    for entry in sorted(PROBLEMS.keys()):
        print(f"- {entry}")

def unpickle_bound(pickled):
    if isinstance(pickled, list):
        return pickled
    return pickle.loads(base64.b64decode(pickled))

def load_result_and_bounds(filename):
    data = json.load(open(filename))
    data["bounds"] = [{"timing": x["timing"], "bound": unpickle_bound(x["bound"])} for x in data["bounds"]]
    return data

def is_all_leq_or_close(a, b):
    return ((a <= b) | (np.isclose(a, b, rtol=1e-4, atol=1e-6))).all()

ErrorMeasure = namedtuple("Error", ["abs", "availability"])
def compute_errors(problem, result, truth, space=None, min_v=None, max_v=None):
    if space is None:
        space = np.linspace(*problem.range, truth.shape[0])
    bounds = [x["bound"](space) if x["bound"] is not None else Error()(space) for x in result["bounds"]]

    if len(bounds) == 0:
        bounds = [Error()(space)]

    if result["bound_type"] == "lb":
        bounds = np.max(np.nan_to_num(bounds, nan=-np.inf), axis=0)

        invalid_part = np.isnan(bounds) | np.isinf(bounds) | np.isnan(truth) | np.isinf(truth)
        if not is_all_leq_or_close(bounds[~invalid_part], truth[~invalid_part]):
            print(f"Lower bound violation detected {problem.name}")
            assert False
    else:
        bounds = np.min(np.nan_to_num(bounds, nan=np.inf), axis=0)

        invalid_part = np.isnan(bounds) | np.isinf(bounds) | np.isnan(truth) | np.isinf(truth)
        if not is_all_leq_or_close(truth[~invalid_part], bounds[~invalid_part]):
            print(f"Upper bound violation detected {problem.name}")
            assert False

    if min_v is None:
        min_v = truth.min()
    if max_v is None:
        max_v = truth.max()
    
    return compute_errors_from_values(bounds, truth, min_v, max_v)

def compute_errors_from_values(values, truth, min_v, max_v):
    truth = truth - min_v
    values = values - min_v
    
    if max_v - min_v > 1e-8:
        truth = truth / (max_v - min_v)
        values = values / (max_v - min_v)

    truth = truth + 1.0
    values = values + 1.0

    error = truth - values

    isinf = np.isinf(error)
    availability = (error.shape[0] - isinf.sum()) / error.shape[0]

    error = error[~isinf]
    rmse = np.sqrt(np.average(error ** 2))
    return ErrorMeasure(rmse, availability)
