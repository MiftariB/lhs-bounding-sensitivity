"""
Bounds on the primary problem.

The secondary problem is the following one:

f(l) =
    min  c^T x
    s.t. A_1 x <= b_1
         (A_2 + l D) x <= b_2

Any bound in this package are in the following form (here for a bound called fct)

fct.ub(problem, lbd_1, lbd_2, *args, **kwargs)
fct.lb(problem, lbd_1, lbd_2, *args, **kwargs)

The bound computes things and then return another function that can be called with the space of l to evaluate.
Typically, the bound is computed between l=lbd_1 and l=lbd_2. If a bound needs a single value of l to work with,
then it uses only lbd_1.
"""
import itertools
import time
from typing import Callable

from bounds.bound_utils import EvaluableBound, get_logger
from problems import Problem

PrimaryBound = Callable[[Problem, float, float, ...], EvaluableBound]

def multi_bound(fct):
    return fct


def single_bound(fct):
    def wrap(what):
        def wrapper(problem, lbds, do_log=False):
            start_time = time.time()
            out = []
            log = get_logger(do_log)
            for idx, (lbd_1, lbd_2) in enumerate(itertools.pairwise(lbds)):
                log(f"Computing bound n°{idx+1}/{len(lbds)-1}")
                bound_start_time = time.time()
                bound = what(problem, lbd_1, lbd_2)
                bound_end_time = time.time()
                out.append({"timing": bound_end_time - bound_start_time, "bound": bound})
            return {
                "timing": time.time() - start_time,
                "bounds": out
            }
        return wrapper
    main = wrap(fct)
    main.ub = wrap(fct.ub)
    main.lb = wrap(fct.lb)
    main.orig = fct
    main.orig_ub = fct.ub
    main.orig_lb = fct.lb

    return main


def upper_bound(fct):
    """ Decorator for a function that computes a bound. The first argument of the function must be the problem
    being solved.

    Adds a .ub method that calls the function itself unchanged,
    and a .lb that calls the function with the dual of the first argument.
    """
    fct.ub = fct
    fct.lb = lambda problem, *args, **kwargs: fct(problem.dual(), *args, **kwargs)
    return fct


def lower_bound(fct):
    """ Decorator for a function that computes a bound. The first argument of the function must be the problem
        being solved.

        Adds a .lb method that calls the function itself unchanged,
        and a .ub that calls the function with the dual of the first argument.
    """
    fct.lb = fct
    fct.ub = lambda problem, *args, **kwargs: fct(problem.dual(), *args, **kwargs)
    return fct


def perfect_bound(fct):
    """ Decorator for a function that computes a bound. The first argument of the function must be the problem
        being solved. Adds a .lb and .ub to be compatible with the @upper_bound and @lower_bound decorators,
        but does not change the argument to the function.
    """
    fct.lb = fct
    fct.ub = fct
    return fct