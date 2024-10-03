from itertools import pairwise

import seaborn as sns
import numpy as np

from bounds.primary.truth import truth
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def show_bounds(problem, limits, upper, lower, precision=1000):
    palette = sns.color_palette("bright", len(upper) + len(lower) + 2)

    ub = np.linspace(1, 1, precision) * np.inf
    lb = np.linspace(1, 1, precision) * -np.inf
    bx = np.linspace(*problem.range, precision)

    by = truth(problem, *problem.range)(bx)
    max_ub = by.max()
    min_lb = by.min()

    fig = sns.lineplot(x=bx, y=by, color=palette[0])

    for left, right in limits:
        for idx, (name, fct) in enumerate(upper):
            by = fct.ub(problem, left, right)(bx)
            if by is None:
                print(f"No bounds between [{left},{right}] for bound {name}")
                continue

            mask = ~np.isnan(by)
            ub[mask] = np.minimum(ub[mask], by[mask])
            sns.lineplot(x=bx, y=by, axes=fig, color=palette[2 + idx], linestyle='--')

        for idx, (name, fct) in enumerate(lower):
            by = fct.lb(problem, left, right)(bx)
            if by is None:
                print(f"No bounds between [{left},{right}] for bound {name}")
                continue

            mask = ~np.isnan(by)
            lb[mask] = np.maximum(lb[mask], by[mask])
            sns.lineplot(x=bx, y=by, axes=fig, color=palette[2 + idx + len(upper)], linestyle='-.')

    fig.axes.fill_between(bx, lb, ub, color=palette[1], alpha=0.2)

    max_ub = max(max_ub, ub.max())
    min_lb = min(min_lb, lb.min())

    if not np.isinf(min_lb) and not np.isinf(max_ub):
        delta = max_ub - min_lb
        fig.set_ylim(min_lb - 0.2 * delta, max_ub + 0.2 * delta)

    legend_elements = [
      Line2D([0], [0], color=palette[2 + idx], label=name, linestyle='--')
      for idx, (name, _) in enumerate(upper)
    ] + [
      Line2D([0], [0], color=palette[2 + idx + len(upper)], label=name, linestyle='-.')
      for idx, (name, _) in enumerate(lower)
    ] + [
      Line2D([0], [0], color=palette[0], label="Truth"),
      Patch(facecolor=palette[1], label='Bounds', alpha=0.2)
    ]

    for elem in set([e for f in limits for e in f]):
        fig.axvline(elem, 0, 1, linestyle="dotted", color="grey")

    fig.get_figure().legend(handles=legend_elements, bbox_to_anchor=(1.00, 1),
                            loc='upper left', borderaxespad=0.)
    return fig


def show_bounds_nb(problem, nb, upper, lower, start=None, end=None, precision=100):
    return show_bounds(problem, list(pairwise(np.linspace(start or problem.range[0], end or problem.range[1], nb + 1))),
                       upper, lower, precision)