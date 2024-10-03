from collections import namedtuple

import numpy as np

from bounds.bound_utils import Error
from problems import solve
import heapq
import time

def max_difference_between_bounds(start, end, upper_bounds, lower_bounds):
    if len(upper_bounds) == 0 or len(lower_bounds) == 0:
        return np.inf
    # TODO: this is wrong and ugly, but works 99.99999999% of the time
    bx = np.linspace(start, end, 1000)
    min_upper_bound = np.minimum.reduce([np.nan_to_num(fct(bx), nan=np.inf) for fct in upper_bounds])
    max_lower_bound = np.maximum.reduce([np.nan_to_num(fct(bx), nan=-np.inf) for fct in lower_bounds])
    return (min_upper_bound - max_lower_bound).max()


def iterative_refining(problem, upper_bounds, lower_bounds, max_y_delta, min_x_delta=-100):
    """
    Iteratively refine upper/lower bounds until the difference between the lower and upper bounds is small enough (less than max_y_delta) or until the difference between
    the two end of the range being considered is less than min_x_delta.

    Note that if min_x_delta is negative, min_x_delta is set to (problem.range[1] - problem.range[0])/-min_x_delta.
    """
    if min_x_delta < 0:
        min_x_delta = (problem.range[1] - problem.range[0]) / -min_x_delta

    def refine(lbd_l, obj_l, lbd_r, obj_r):
        ub = [z["bound"] for fct in upper_bounds for y in [fct.ub(problem, [lbd_l, lbd_r])] if y is not None for z in y["bounds"] if not isinstance(z["bound"], Error)]
        lb = [z["bound"] for fct in lower_bounds for y in [fct.lb(problem, [lbd_l, lbd_r])] if y is not None for z in y["bounds"] if not isinstance(z["bound"], Error)]

        if len(ub) == 0 or len(lb) == 0 and min_x_delta == 0:
            raise Exception(
                "No bounds found but no threshold for stopping: infinite loop prevented by raising this exception")

        if lbd_r - lbd_l > min_x_delta and max_difference_between_bounds(lbd_l, lbd_r, ub, lb) > max_y_delta:
            # refine, let's cut in half
            lbd_m = (lbd_r + lbd_l) / 2
            obj_m = solve(problem, lbd_m)
            yield from refine(lbd_l, obj_l, lbd_m, obj_m)
            yield from refine(lbd_m, obj_m, lbd_r, obj_r)
        else:
            yield (lbd_l, obj_l, lbd_r, obj_r, ub, lb)

    return list(refine(problem.range[0], solve(problem, problem.range[0]), problem.range[1], solve(problem, problem.range[1])))

Interval = namedtuple("Interval", ["neg_error", "lbd_l", "lbd_r", "ub", "lb", "gen_time"])
Point = namedtuple('Point', ['lbd', 'obj', "lbd_l", "lbd_r", 'gen_time'])

def iterative_refining_wfs(problem, upper_bounds, lower_bounds, max_y_delta, min_x_delta=-1000,
                           ground_truth_between_intervals=False, timelimit=float("inf")):
    """
    Iteratively refine upper/lower bounds until the difference between the lower and upper bounds is small enough (less than max_y_delta) or until the difference between
    the two end of the range being considered is less than min_x_delta.

    Note that if min_x_delta is negative, min_x_delta is set to (problem.range[1] - problem.range[0])/-min_x_delta.

    This function uses a "worst-first-search" technique: among all the currently available sections, it
    refines the one with the worst distance between the bounds
    """
    if min_x_delta < 0:
        min_x_delta = (problem.range[1] - problem.range[0]) / -min_x_delta
    if min_x_delta == 0:
        raise Exception("Please choose a non-zero min_x_delta. Running with 0 would lead to infinite loops.")

    todo = []
    done = []
    gt_points = []

    start_time = time.time()

    def compute_interval(lbd_l, lbd_r):
        if lbd_r - lbd_l <= min_x_delta:
            mid = (lbd_r + lbd_l)/2
            gt_points.append(Point(mid, solve(problem, mid), lbd_l, lbd_r, time.time() - start_time))
            return

        ub = [z["bound"] for fct in upper_bounds for y in [fct.ub(problem, [lbd_l, lbd_r])] if y is not None for z in
              y["bounds"] if not isinstance(z["bound"], Error) and z['bound'] is not None]
        lb = [z["bound"] for fct in lower_bounds for y in [fct.lb(problem, [lbd_l, lbd_r])] if y is not None for z in
              y["bounds"] if not isinstance(z["bound"], Error) and z['bound'] is not None]

        interval = Interval(
            -max_difference_between_bounds(lbd_l, lbd_r, ub, lb),
            float(lbd_l), float(lbd_r),
            ub, lb,
            time.time() - start_time
        )
        heapq.heappush(todo, interval)

    compute_interval(*problem.range)

    while len(todo) and time.time()-start_time < timelimit:
        interval: Interval = heapq.heappop(todo)
        done.append(interval)
        if -interval.neg_error < max_y_delta:
            continue
        mid = (interval.lbd_r + interval.lbd_l) / 2

        compute_interval(interval.lbd_l, mid)

        if time.time()-start_time >= timelimit:
            break

        compute_interval(mid, interval.lbd_r)

    return done + todo, gt_points

if __name__ == '__main__':
    from example_problems.random_1 import problem as random_1
    from bounds.primary.lagrangian import bound_lagrangian_flat
    from bounds.primary.robust import bound_robust_flat, bound_robust_line_left, bound_robust_line_right
    from iterative_refining.plot_utils import display_refining, display_refining_wfs

    bounds, gt_points = iterative_refining_wfs(random_1, [
        bound_robust_flat,
        bound_robust_line_left,
        bound_robust_line_right,
        bound_lagrangian_flat
    ], [
        bound_robust_flat,
        bound_robust_line_left,
        bound_robust_line_right,
        bound_lagrangian_flat
    ], 0.1, -1000)

    display_refining_wfs(bounds, gt_points)