import base64

from bounds.primary.coefficient_bound import coefficient_flat
from bounds.primary.robust import bound_robust_flat, bound_robust_fixed_slope_pairwise
from bounds.primary.robust import bound_robust_line_left
from bounds.primary.robust import bound_robust_line_right
from bounds.primary.robust import bound_robust_xyflat
from bounds.primary.lagrangian import bound_lagrangian_bisegment, bound_lagrangian_bisegment_coef, bound_lagrangian_bisegment_coef_adv, bound_lagrangian_flat, bound_lagrangian_flat_coef, bound_lagrangian_flat_coef_adv, bound_lagrangian_line, bound_lagrangian_quadratic, \
    bound_lagrangian_envelope
from bounds.primary.robust import robust_concave_envelope

from iterative_refining.primary import iterative_refining_wfs
from bounds.primary.lagrangian import bound_lagrangian_flat, bound_lagrangian_envelope
from bounds.primary.robust import bound_robust_flat, bound_robust_line_left, bound_robust_line_right


import multiprocessing
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
    "lagrangian_flat_coef": bound_lagrangian_flat_coef,
    "lagrangian_flat_coef_adv": bound_lagrangian_flat_coef_adv,
    "lagrangian_bisegment": bound_lagrangian_bisegment,
    "lagrangian_bisegment_coef": bound_lagrangian_bisegment_coef,
    "lagrangian_bisegment_coef_adv": bound_lagrangian_bisegment_coef_adv,
    "component_flat": coefficient_flat,
    "lagrangian_quadratic": bound_lagrangian_quadratic,
    "lagrangian_line": bound_lagrangian_line,
    "robust_concave_envelope": robust_concave_envelope,
    "robust_fixed_slope_pairwise": bound_robust_fixed_slope_pairwise,
    "lagrangian_envelope": bound_lagrangian_envelope
}


def solve(lower_bounds, up_bounds, problem, output, timelimit, min_x_delta):
    all_upper_bounds_required = [bounds[bnd] for bnd in up_bounds]
    all_lower_bounds_required = [bounds[bnd] for bnd in lower_bounds]

    out_bounds, gt_points = iterative_refining_wfs(problem, all_upper_bounds_required, all_lower_bounds_required,
                                                   0, min_x_delta=min_x_delta, timelimit=timelimit)
    output["bounds"] = base64.b64encode(pickle.dumps(out_bounds)).decode("ascii")
    output["gt_points"] = base64.b64encode(pickle.dumps(gt_points)).decode("ascii")

    return output, out_bounds, gt_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description='experiments robust var')

    parser.add_argument("problem", help="Problem name", choices=PROBLEMS.keys())
    parser.add_argument("timelimit", help="Time limit in seconds", type=int)
    parser.add_argument("truth", help="Path to the truth json file", type=str)
    parser.add_argument("pointlimit", help="Time limit in gt points", type=int)
    parser.add_argument("output", help="Output filename", type=str)
    parser.add_argument("--solo", help="Considers only one bound as upper bound and the other as lower bound. "
                                       "Expects two arguments.", action='store_true')
    #parser.add_argument("output_video", help="Images directory", type=str)
    parser.add_argument("bounds", help="Bounds name", choices=bounds.keys(), nargs='+')
    parser.add_argument("--min_x_delta", help="Minimum x delta", type=float, default=-100)

    _args = parser.parse_args()
    _output = {
        "name": _args.bounds,
        "problem": _args.problem,
        "bounds": [],
        "gt_points": []
    }

    problem = PROBLEMS[_args.problem]()

    gt_time = (json.load(open(_args.truth, 'r'))["timing"]/100.)
    timelimit = min(_args.timelimit, _args.pointlimit * gt_time)

    pool = multiprocessing.Pool(processes=1)
    if _args.solo:
        res = pool.apply_async(solve, ([_args.bounds[0]], [_args.bounds[1]], problem, _output, timelimit, _args.min_x_delta))
    else:
        res = pool.apply_async(solve, (set(_args.bounds), set(_args.bounds), problem, _output, timelimit, _args.min_x_delta))

    try:
        _output, out_bounds, gt_points = res.get(timeout=timelimit*2)
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
