from bounds.primary.truth import truth
from example_problems.microgrid_small import problem as microgrid_small
from example_problems.random_1 import problem as toy_1
from example_problems.random_2 import problem as toy_2
from example_problems.random_3 import problem as toy_3
from example_problems.three_clusters import problem as belgian_model
from example_problems.three_month_three_clusters  import problem as belgian_model_small

from expe_lib import PROBLEMS

import numpy as np
import time
import argparse
import json

from expe_lib import PROBLEMS

parser = argparse.ArgumentParser(allow_abbrev=False,
                                 description='experiments robust var')

parser.add_argument("nb_bounds", help="Number of bounds", type=int, default=100)
parser.add_argument("problem", help="Problem name", type=str, choices=PROBLEMS.keys())
parser.add_argument("output_json", help="Output filename", type=str)
parser.add_argument("output_npz", help="Output filename", type=str)

args = parser.parse_args()
filename_json = args.output_json
filename_npz = args.output_npz
N = args.nb_bounds

problem = PROBLEMS[args.problem]()

space = np.linspace(*problem.range, N)

start_time = time.time()
out = truth(problem, [space[0], space[-1]])["bounds"][0]["bound"](space)
end_time = time.time() - start_time

np.savez(filename_npz, out)

output = {
    "name": "truth",
    "N": N,
    "dualize": False,
    "problem": args.problem,
    "timing - fixed costs": 0,
    "bounds": [],
    "timing": end_time
}

with open(filename_json, "w") as outfile:
    json.dump(output, outfile)
