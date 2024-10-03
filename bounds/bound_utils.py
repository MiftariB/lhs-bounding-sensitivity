from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any, Optional, Tuple
import numpy as np
import numpy.typing as npt

# function that compute bounds typically return an object that can return the bound
# when evaluated on any value of l.
EvaluableBound = Callable[[npt.ArrayLike], npt.ArrayLike]

# The dataclasses below are fake functions that respect the EvaluableBound contract.

@dataclass
class LimitedSpace:
    """ Represent a function over a range """
    fct: Callable[..., Any]
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, self.fct)


@dataclass
class Constant:
    """ A constant over a range """
    value: float
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, lambda s: s * 0 + self.value)


@dataclass
class Line:
    """ An affine function over a range """
    a: float
    b: float
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, lambda s: self.a * s + self.b)


@dataclass
class Quadratic:
    """ A quadratic function over a range """
    a: float
    b: float
    c: float
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, lambda s: self.a * s * s + self.b * s + self.c)

@dataclass
class Maximize:
    """ A quadratic function over a range """
    objects: any
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, lambda s: np.array([x(s) for x in self.objects]).max(axis=0))
        #return np.max(np.array([_limit_space_and_run(space, self.limits, f) for f in self.objects]), axis=1)

@dataclass
class Minimize:
    """ A quadratic function over a range """
    objects: any
    limits: Optional[Tuple[float, float]]

    def __call__(self, space):
        return _limit_space_and_run(space, self.limits, lambda s: np.array([x(s) for x in self.objects]).min(axis=0))


@dataclass
class Error:
    """ An error """
    def __call__(self, space):
        return np.full_like(space, float("nan"))


def _limit_space_and_run(space, limits, fct):
    out = np.full_like(space, float("nan"))
    mask = (limits[0] <= space) & (space <= limits[1])
    out[mask] = fct(space[mask])
    return out

def get_logger(enable_log):
    if not enable_log:
        def f(*_1, **_2):
            return
    else:
        def f(*args, **kwargs):
            print(f"[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}]", *args, **kwargs)
    return f