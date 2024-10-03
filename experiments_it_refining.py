import base64
import sys

from bounds.primary.robust import bound_robust_flat, bound_robust_fixed_slope_pairwise
from bounds.primary.robust import bound_robust_line_left
from bounds.primary.robust import bound_robust_line_right
from bounds.primary.robust import bound_robust_xyflat
from bounds.primary.lagrangian import bound_lagrangian_flat, bound_lagrangian_line, bound_lagrangian_quadratic, \
    bound_lagrangian_envelope
from bounds.primary.robust import robust_concave_envelope

from iterative_refining.primary import iterative_refining, iterative_refining_wfs, Point
from iterative_refining.plot_utils import display_refining, display_refining_wfs
from bounds.primary.lagrangian import bound_lagrangian_flat, bound_lagrangian_envelope
from bounds.primary.robust import bound_robust_flat, bound_robust_line_left, bound_robust_line_right
import imageio.v3 as iio
import matplotlib.pyplot as plt


import multiprocessing
import numpy as np
import argparse
import json
import pickle
import tempfile

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


def solve(up_bounds, lower_bounds, problem, output, timelimit):
    all_upper_bounds_required = [bounds[bnd] for bnd in up_bounds]
    all_lower_bounds_required = [bounds[bnd] for bnd in lower_bounds]

    out_bounds, gt_points = iterative_refining_wfs(problem, all_upper_bounds_required, all_lower_bounds_required,
                                                   0, min_x_delta=-100, timelimit=timelimit)
    output["bounds"] = base64.b64encode(pickle.dumps(out_bounds)).decode("ascii")
    output["gt_points"] = base64.b64encode(pickle.dumps(gt_points)).decode("ascii")

    return output, out_bounds, gt_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description='experiments robust var')

    parser.add_argument("problem", help="Problem name", choices=PROBLEMS.keys())
    parser.add_argument("timelimit", help="Time limit in seconds", type=int)
    parser.add_argument("output", help="Output filename", type=str)
    parser.add_argument("--solo", help="Considers only one bound as upper bound and the other as lower bound. "
                                       "Expects two arguments.", action='store_true')
    #parser.add_argument("output_video", help="Images directory", type=str)
    parser.add_argument("bounds", help="Bounds name", choices=bounds.keys(), nargs='+')

    _args = parser.parse_args()
    _output = {
        "name": _args.bounds,
        "problem": _args.problem,
        "bounds": [],
        "gt_points": []
    }

    problem = PROBLEMS[_args.problem]()

    pool = multiprocessing.Pool(processes=1)
    if _args.solo:

        res = pool.apply_async(solve, ([_args.bounds[0]], [_args.bounds[1]], problem, _output, _args.timelimit))
    else:
        res = pool.apply_async(solve, (set(_args.bounds), set(_args.bounds), problem, _output, _args.timelimit))

    try:
        _output, out_bounds, gt_points = res.get(timeout=_args.timelimit*2)
    except multiprocessing.TimeoutError:
        print("---------")
        print("TIMEOUT")
        print("---------")
        _output["timing"] = _args.timelimit
        out_bounds = []
        gt_points = []

    with open(_args.output, "w") as outfile:
        json.dump(_output, outfile)

    pool.terminate()

    #bounds = sorted(out_bounds + gt_points, key=lambda x: x.gen_time)
    #seen_gt_points = []

    #yrange = min(x.obj for x in gt_points), max(x.obj for x in gt_points)

    #with tempfile.TemporaryDirectory() as tmpdirname:
    #    for i, bnd in enumerate(bounds):
    #        fig = plt.figure()
    #        if isinstance(bnd, Point):
    #            seen_gt_points.append(bnd)
    #        display_refining_wfs(problem.range, bounds[:i + 1], seen_gt_points, yrange)
    #        fig.savefig(f"{tmpdirname}/{i}.jpg")
    #        plt.close(fig)

    #    frames = np.stack([iio.imread(f"{tmpdirname}/{i}.jpg") for i, _ in enumerate(bounds)], axis=0)
    #iio.imwrite(_args.output_video, frames, fps=10)