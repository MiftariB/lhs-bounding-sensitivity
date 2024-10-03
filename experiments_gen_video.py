from celluloid import Camera

from iterative_refining.primary import Point
from iterative_refining.plot_utils import display_refining_wfs
import imageio.v3 as iio
import matplotlib.pyplot as plt


import numpy as np
import argparse
import json
import tempfile

from expe_lib import unpickle_bound, PROBLEMS


def load_it_refining_result(filename):
    data = json.load(open(filename))
    data["bounds"] = unpickle_bound(data["bounds"])
    data["gt_points"] = unpickle_bound(data["gt_points"])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description='experiments robust var')

    parser.add_argument("truth_file", help="Truth file path", type=str)
    parser.add_argument("it_file", help="Iterative refining filename", type=str)
    parser.add_argument("output_video", help="Output filename", type=str)

    _args = parser.parse_args()

    truth_data = np.load(_args.truth_file)["arr_0"]
    it_data = load_it_refining_result(_args.it_file)

    out_bounds = it_data["bounds"]
    gt_points = it_data["gt_points"]
    problem_name = it_data["problem"]
    problem = PROBLEMS[problem_name]()

    bounds = sorted(out_bounds + gt_points, key=lambda x: x.gen_time)
    seen_gt_points = []

    delta = max(truth_data) - min(truth_data)
    yrange = min(truth_data)-delta/2, max(truth_data)+delta/2

    fig = plt.Figure()
    camera = Camera(fig)
    for i, bnd in enumerate(bounds):
        print(f"- Image n°{i+1}/{len(bounds)}")
        if isinstance(bnd, Point):
            seen_gt_points.append(bnd)
        display_refining_wfs(problem.range, bounds[:i + 1], seen_gt_points, yrange, fig=fig)
        camera.snap()
    print(f"- Gen animation")
    animation = camera.animate()
    print(f"- Output")
    animation.save(_args.output_video)