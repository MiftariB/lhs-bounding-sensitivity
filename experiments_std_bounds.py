import base64
import sys

from bounds.primary.robust import bound_robust_flat, bound_robust_fixed_slope_pairwise
from bounds.primary.robust import bound_robust_line_left
from bounds.primary.robust import bound_robust_line_right
from bounds.primary.robust import bound_robust_xyflat
from bounds.primary.lagrangian import bound_lagrangian_flat, bound_lagrangian_line, bound_lagrangian_quadratic, \
    bound_lagrangian_envelope
from bounds.primary.robust import robust_concave_envelope

import multiprocessing
import numpy as np
import argparse
import json
import pickle

from expe_lib import PROBLEMS

bounds = {
    "robust_flat": bound_robust_flat,
    "robust_line_left": bound_robust_line_left,
    "robust_line_right": bound_robust_line_right,
    "robust_xyflat": bound_robust_xyflat,
    "lagrangian_flat": bound_lagrangian_flat,
    "lagrangian_quadratic": bound_lagrangian_quadratic,
    "lagrangian_line": bound_lagrangian_line,
    "robust_concave_envelope": robust_concave_envelope,
    "robust_fixed_slope_pairwise": bound_robust_fixed_slope_pairwise,
    "lagrangian_envelope": bound_lagrangian_envelope
}


def solve(N, bound_name, bound_type, problem_name, output):
    bnd = bounds[bound_name]
    problem = PROBLEMS[problem_name]()
    space = np.linspace(*problem.range, N)

    if bound_type == "ub":
        result = bnd.ub(problem, space, do_log=True)
    elif bound_type == "lb":
        result = bnd.lb(problem, space, do_log=True)
    elif bound_type == "primal":
        result = bnd(problem, space, do_log=True)
    else:
        result = bnd(problem.dual(), space, do_log=True)

    output["timing"] = result["timing"]
    output["bounds"] = [
        {
            "timing": x["timing"],
            "bound": base64.b64encode(pickle.dumps(x["bound"])).decode("ascii")
        }
        for x in result["bounds"]
    ]
    output["timing - fixed costs"] = result["timing"] - sum(x["timing"] for x in output["bounds"])

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description='experiments robust var')



    parser.add_argument("bound_type", help="Type of bound to obtain", choices=['ub', 'lb', 'primal', 'dual'])
    parser.add_argument("nb_bounds", help="Number of bounds", type=int)
    parser.add_argument("problem", help="Problem name", choices=PROBLEMS.keys())
    parser.add_argument("bound", help="Bound name", choices=bounds.keys())
    parser.add_argument("timelimit", help="Time limit in seconds", type=int)
    parser.add_argument("output", help="Output filename", type=str)

    _args = parser.parse_args()

    _output = {
        "name": _args.bound,
        "N": _args.nb_bounds,
        "bound_type": _args.bound_type,
        "problem": _args.problem,
        "timing - fixed costs": 0,
        "bounds": []
    }

    pool = multiprocessing.Pool(processes=1)
    res = pool.apply_async(solve, (_args.nb_bounds, _args.bound, _args.bound_type, _args.problem, _output))

    try:
        _output = res.get(timeout=_args.timelimit)
    except multiprocessing.TimeoutError:
        print("---------")
        print("TIMEOUT")
        print("---------")
        _output["timing"] = _args.timelimit

    with open(_args.output, "w") as outfile:
        json.dump(_output, outfile)

    pool.terminate()