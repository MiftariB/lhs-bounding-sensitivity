import numpy as np
import seaborn as sns

from iterative_refining.primary import Interval


def display_refining(refining, precision=1000):
    out_x = np.array([x[0] for x in refining] + [refining[-1][2]])
    out_y = np.array([x[1] for x in refining] + [refining[-1][3]])

    fig = sns.scatterplot(x=out_x, y=out_y)

    for lbd_l, obj_l, lbd_r, obj_r, ubs, lbs in refining:
        ub = None
        lb = None
        bx = np.linspace(lbd_l, lbd_r, precision)
        if len(ubs):
            ub = np.minimum.reduce([x(bx) for x in ubs])
        if len(lbs):
            lb = np.maximum.reduce([x(bx) for x in lbs])
        if ub is not None and lb is not None:
            fig.axes.fill_between(bx, lb, ub, alpha=0.3, color="green")

    sns.lineplot(x=out_x, y=out_y, linestyle="dashed")

def display_refining_wfs(xrange, refining, gt_points, init_obj_range, precision=10000, fig=None):
    """
    :param xrange:
    :param refining: may contain Interval and Point objects
    :param gt_points: only Point objects
    :param precision:
    :return:
    """
    sns.set_palette(sns.color_palette(["black"]))
    fig = sns.scatterplot(x=[x.lbd for x in gt_points], y=[x.obj for x in gt_points], ax=fig.gca())

    min_v, max_v = xrange
    #fig.set_xlim(min_v, max_v)

    global_x = np.linspace(min_v, max_v, precision)
    global_ub = np.zeros_like(global_x)
    global_ub[:] = np.inf
    global_lb = np.zeros_like(global_x)
    global_lb[:] = -np.inf

    for entry in refining:
        if isinstance(entry, Interval):
            _, lbd_l, lbd_r, ubs, lbs, _ = entry
            if len(ubs):
                global_ub = np.minimum.reduce([np.nan_to_num(x(global_x), nan=np.inf) for x in ubs] + [global_ub])
            if len(lbs):
                global_lb = np.maximum.reduce([np.nan_to_num(x(global_x), nan=-np.inf) for x in lbs] + [global_lb])
        else:
            _, val, lbd_l, lbd_r, _ = entry
            arr_range = (global_x >= lbd_l) & (global_x <= lbd_r)
            global_ub[arr_range] = np.minimum(global_ub[arr_range], val)
            global_lb[arr_range] = np.maximum(global_lb[arr_range], val)

    fig.axes.fill_between(global_x, global_lb, global_ub, alpha=0.3, color="green")

    #fig.set_ylim(*init_obj_range)

    return fig