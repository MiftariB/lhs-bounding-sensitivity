from bounds.bound_utils import LimitedSpace
from bounds.primary import perfect_bound, single_bound
from problems import solve

import numpy as np


@single_bound
@perfect_bound
def truth(problem, lbd_1, lbd_2, maxtime=None):
    return LimitedSpace(lambda space: np.array([solve(problem, i, maxtime=maxtime, fail_on_timeout=True) for i in space]), (lbd_1, lbd_2))