import itertools
import time
import numpy as np
import scipy.sparse as sp

from bounds.bound_utils import Error, Constant, Line, Quadratic, Maximize, get_logger, Minimize
from bounds.primary import lower_bound, multi_bound
from problems import Problem_sparse
from solveapi import solve_api, NoSolution


def get_sol_and_alpha(problem, lbd):
    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1])

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    api.add_constr((a_2_eq + lbd * d_eq), "==", b_2_eq.reshape(-1, ))
    api.add_constr((a_2_ineq + lbd * d_ineq), "<=", b_2_ineq.reshape(-1, ))

    concerned_eq = slice(a_1_eq.shape[0] + a_1_ineq.shape[0],
                         a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0])

    concerned_ineq = slice(a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0],
                           a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0] + a_2_ineq.shape[0])

    api.set_obj(c.transpose(), "minimize" if mini else "maximize")
    api.optimize()

    status = api.get_status()
    if status == "unknown":
        return None
    dual = api.get_dual()
    return api.get_objective(), dual[concerned_eq], dual[concerned_ineq]


def get_sol_for_alpha(problem, lbd, alpha_eq, alpha_ineq):
    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem
    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1])
    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    api.set_obj(c.transpose()-(alpha_eq.transpose() @ (a_2_eq + lbd * d_eq))
                - (alpha_ineq.transpose() @ (a_2_ineq + lbd * d_ineq)),
                "minimize" if mini else "maximize")

    api.optimize()
    status = api.get_status()
    if status == "unknown":
        return None
    return api.get_objective() + (alpha_eq.transpose() @ b_2_eq) + (alpha_ineq.transpose() @ b_2_ineq)

@multi_bound
@lower_bound
def bound_lagrangian_envelope(problem, lbds, do_log=False):
    assert isinstance(problem, Problem_sparse)
    nb_var = problem.a_1_eq.shape[1]
    mini = problem.minimize

    # get ground truth and duals
    log = get_logger(do_log)
    start = time.time()

    gt_points = []
    for idx, lbd in enumerate(lbds[:-1]):
        log(f"Solving 'ground truth problem' n°{idx + 1}/{len(lbds) - 1}")
        gt_points.append(get_sol_and_alpha(problem, lbd))

    log("Done computing ground truth")

    api = solve_api("cplex")
    api.activate_crossover()
    api.add_var(nb_var)
    api.add_constr(problem.a_1_eq, '==', problem.b_1_eq.reshape(-1))
    api.add_constr(problem.a_1_ineq, '<=', problem.b_1_ineq.reshape(-1))

    def gen_bounds(start_lbd, rho_eq, rho_ineq):
        l_c = problem.c - (rho_eq.transpose() @ problem.a_2_eq + rho_ineq.transpose() @ problem.a_2_ineq).reshape(
            (-1, 1))
        l_bias = rho_eq.transpose() @ problem.b_2_eq + rho_ineq.transpose() @ problem.b_2_ineq
        l_cl = -(rho_eq.transpose() @ problem.d_eq + rho_ineq.transpose() @ problem.d_ineq).reshape((-1, 1))

        def solve_for_lbd(cur_lbd):
            api.set_obj((l_c + l_cl * cur_lbd).astype(float), "minimize" if mini else "maximize")

            api.optimize()

            status = api.get_status()
            if status == "unknown":
                # numerical problems
                log("Encountered numerical problems")
                return Error()

            var_basis, slack_basis = api.get_basis()

            nb_var_basis = np.sum(var_basis)
            nb_var_non_basis = nb_var - nb_var_basis
            nb_basis_slack = np.sum(slack_basis)
            nb_non_basis_slack = len(slack_basis) - nb_basis_slack

            all_basis = np.concatenate([var_basis, slack_basis])

            r_nb = np.zeros((nb_var_non_basis + nb_non_basis_slack,), dtype=np.float64)
            r_nb[0:nb_var_non_basis] = l_cl[~var_basis, 0]

            r_b = np.zeros(api.model.linear_constraints.get_num(), dtype=np.float64)
            r_b[0:nb_var_basis] = l_cl[var_basis, 0]

            r_b_M_b_m1 = np.array(api.model.solution.advanced.btran(r_b))

            slacked_matrix_eq = sp.hstack([problem.a_1_eq,
                                           0 * sp.eye(problem.a_1_eq.shape[0],
                                                      n=api.model.linear_constraints.get_num())],
                                          format="csr")

            slacked_matrix_ineq = sp.hstack([problem.a_1_ineq,
                                             sp.eye(problem.a_1_ineq.shape[0],
                                                    n=api.model.linear_constraints.get_num(),
                                                    k=slacked_matrix_eq.shape[0])], format="csr")
            big_M_matrix = sp.vstack([slacked_matrix_eq, slacked_matrix_ineq], format="csr")
            r_mul = r_b_M_b_m1 @ big_M_matrix[:, np.logical_not(all_basis)]

            M_total = r_nb - r_mul

            c_obj_full = np.zeros_like(all_basis, dtype=float)
            c_obj_full[0:nb_var] = (l_c + l_cl * cur_lbd).reshape(-1)

            dual_var = np.array(api.model.solution.get_dual_values())
            reduced_costs_non_basic = c_obj_full[np.logical_not(all_basis)] - (dual_var @ big_M_matrix)[
                np.logical_not(all_basis)]

            non_zeros_in_M_total = M_total[np.abs(M_total) > 1e-6]
            non_zeros_in_non_basic_rc = reduced_costs_non_basic[np.abs(M_total) > 1e-6]
            conditions = -np.divide(non_zeros_in_non_basic_rc, non_zeros_in_M_total)

            upper_bnd = np.inf
            lower_bnd = -np.inf

            try:
                if mini:
                    upper_bnd = conditions[non_zeros_in_M_total < 0].min()
                else:
                    lower_bnd = conditions[non_zeros_in_M_total < 0].max()
            except ValueError:
                pass

            try:
                if mini:
                    lower_bnd = conditions[non_zeros_in_M_total > 0].max()
                else:
                    upper_bnd = conditions[non_zeros_in_M_total > 0].min()
            except ValueError:
                pass

            full_sol = api.get_solution().reshape((-1, 1))
            a = l_cl.transpose() @ full_sol
            b = l_c.transpose() @ full_sol + l_bias

            out = Line(a[0, 0], b[0, 0], (cur_lbd + lower_bnd, cur_lbd + upper_bnd))
            log(out)
            return out

        epsilon = max(1e-5, (max(lbds) - min(lbds)) / 100.0)

        # to the right
        bounds_right = [solve_for_lbd(start_lbd)]
        if not isinstance(bounds_right[0], Error):
            while bounds_right[-1].limits[1] < lbds[-1]:
                try:
                    next_line = solve_for_lbd(bounds_right[-1].limits[1] + epsilon)
                    if isinstance(next_line, Error):
                        break
                    bounds_right.append(next_line)
                except:
                    break

            # to the left
            bounds_left = [bounds_right[0]]
            while bounds_left[-1].limits[0] > lbds[0]:
                try:
                    next_line = solve_for_lbd(bounds_left[-1].limits[0] - epsilon)
                    if isinstance(next_line, Error):
                        break
                    bounds_left.append(next_line)
                except:
                    break

            bounds = list(reversed(bounds_left))[:-1] + bounds_right
            if mini:
                return Maximize(bounds, limits=(bounds[0].limits[0], bounds[-1].limits[1]))
            else:
                return Minimize(bounds, limits=(bounds[0].limits[0], bounds[-1].limits[1]))
        else:
            return Error()

    out = []
    for idx, (lbd_1, sol_lbd_1) in enumerate(zip(lbds, gt_points)):
        log(f"Computing bound n°{idx}/{len(lbds)}")
        if sol_lbd_1 is None:
            out.append({
                "bound": Error(),
                "timing": 0.0
            })
        else:
            bound_start = time.time()
            bound = gen_bounds(lbd_1, sol_lbd_1[1], sol_lbd_1[2])
            bound_end = time.time()
            out.append({
                "bound": bound,
                "timing": bound_end - bound_start
            })

    return {
        "timing": time.time() - start,
        "bounds": out
    }

@multi_bound
@lower_bound
def bound_lagrangian_flat(problem, lbds, do_log=False):
    assert isinstance(problem, Problem_sparse)
    log = get_logger(do_log)

    start = time.time()

    gt_points = []
    for idx, lbd in enumerate(lbds):
        log(f"Solving 'ground truth problem' n°{idx}/{len(lbds)}")
        gt_points.append(get_sol_and_alpha(problem, lbd))

    log("Done computing ground truth")


    # instance
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem
    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1])
    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    def get_sol_for_alpha(lbd, alpha_eq, alpha_ineq):
        api.set_obj(c.transpose() - (alpha_eq.transpose() @ (a_2_eq + lbd * d_eq))
                    - (alpha_ineq.transpose() @ (a_2_ineq + lbd * d_ineq)),
                    "minimize" if mini else "maximize")

        api.optimize()
        status = api.get_status()
        if status == "unknown":
            return None
        return api.get_objective() + (alpha_eq.transpose() @ b_2_eq) + (alpha_ineq.transpose() @ b_2_ineq)

    def get_bound(lbd_1, sol_low, alpha_low_eq, alpha_low_ineq, lbd_2, sol_high, alpha_high_eq, alpha_high_ineq):
        bound = float("-inf")
        f = max
        f2 = min

        if not problem.minimize:
            bound = float("inf")
            f = min
            f2 = max

        hl = get_sol_for_alpha(lbd_1, alpha_high_eq, alpha_high_ineq)
        if hl is not None:
            bound = f(bound, f2(hl, sol_high))

        hl = get_sol_for_alpha(lbd_2, alpha_low_eq, alpha_low_ineq)
        if hl is not None:
            bound = f(bound, f2(hl, sol_low))

        if bound != float("inf") and bound != float("-inf"):
            return Constant(bound, (lbd_1, lbd_2))
        else:
            return Error()

    out = []
    for idx, ((lbd_1, sol_lbd_1), (lbd_2, sol_lbd_2)) in enumerate(itertools.pairwise(zip(lbds, gt_points))):
        log(f"Computing bound n°{idx}/{len(lbds)-1}")
        if sol_lbd_1 is None or sol_lbd_2 is None:
            out.append({
                "bound": Error(),
                "timing": 0.0
            })
        else:
            bound_start = time.time()
            bound = get_bound(lbd_1, *sol_lbd_1, lbd_2, *sol_lbd_2)
            bound_end = time.time()
            out.append({
                "bound": bound,
                "timing": bound_end - bound_start
            })

    return {
        "timing": time.time() - start,
        "bounds": out
    }

@multi_bound
@lower_bound
def bound_lagrangian_quadratic(problem, lbds, do_log=False, separate=False):
    assert isinstance(problem, Problem_sparse)
    log = get_logger(do_log)

    start = time.time()

    for idx in range(len(lbds)-1):
        if lbds[idx] < 0 < lbds[idx + 1]:
            log("Adding 0 to the lambdas", idx, lbds[:idx+1], lbds[idx+1:])
            lbds = np.concatenate([lbds[:idx+1], [0.0],lbds[idx+1:]])
            break

    gt_points = []
    for idx, lbd in enumerate(lbds):
        log(f"Solving 'ground truth problem' n°{idx}/{len(lbds)}")
        gt_points.append(get_sol_and_alpha(problem, lbd))

    log("Done computing ground truth")

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    def gen_problem():
        api = solve_api("cplex")
        api.add_var(a_1_eq.shape[1])
        api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
        api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
        return api

    part1 = gen_problem()
    part2 = gen_problem() if separate else part1
    part3 = gen_problem() if separate else part1

    def get_bound(lbd_1, sol_low, alpha_low_eq, alpha_low_ineq, lbd_2, sol_high, alpha_high_eq, alpha_high_ineq):
        e_eq = -(alpha_high_eq - alpha_low_eq) / (lbd_2 - lbd_1)
        e_ineq = -(alpha_high_ineq - alpha_low_ineq) / (lbd_2 - lbd_1)
        f_eq = -(alpha_low_eq * lbd_2 - alpha_high_eq * lbd_1) / (lbd_2 - lbd_1) #alpha_low_eq - e_eq * lbd_1
        f_ineq = -(alpha_low_ineq * lbd_2 - alpha_high_ineq * lbd_1) / (lbd_2 - lbd_1)#alpha_low_ineq - e_ineq * lbd_1

        if lbd_1 <= 0 and lbd_2 <= 0:
            mode = "minimize" if not problem.minimize else "maximize"
        else:
            mode = "minimize" if problem.minimize else "maximize"

        def const_coef():
            model = part1
            model.set_obj(c.transpose() + (f_ineq.transpose() @ a_2_ineq) + (f_eq.transpose() @ a_2_eq),
                          "minimize" if mini else "maximize")
            model.optimize()
            return model.get_objective() - (f_eq.transpose() @ b_2_eq).flatten()[0] - (f_ineq.transpose() @ b_2_ineq).flatten()[0]

        def linear_coef():
            model = part2
            model.set_obj((e_eq.transpose() @ a_2_eq) + (e_ineq.transpose() @ a_2_ineq)
                          + (f_eq.transpose() @ d_eq) + (f_ineq.transpose() @ d_ineq), mode)

            model.optimize()
            return model.get_objective() - (e_eq.transpose() @ b_2_eq).flatten()[0] - (e_ineq.transpose() @ b_2_ineq).flatten()[0]

        def quadratic_coef():
            model = part3
            model.set_obj((e_eq.transpose() @ d_eq) + (e_ineq.transpose() @ d_ineq),
                          "minimize" if mini else "maximize")
            model.optimize()
            return model.get_objective()

        limits = [-np.inf, 0.0] if lbd_1 <= lbd_2 <= 0 else [0.0, np.inf]

        l_right_ineq = -alpha_low_ineq + lbd_1*(alpha_high_ineq-alpha_low_ineq)/(lbd_2-lbd_1)
        l_left_ineq = (alpha_high_ineq-alpha_low_ineq)/(lbd_2-lbd_1)

        try:
            out_c = const_coef()
            out_b = linear_coef()
            out_a = quadratic_coef()

            if (l_left_ineq < -1e-6).any():
                limits[0] = max(limits[0], (l_right_ineq[l_left_ineq < -1e-6] / l_left_ineq[l_left_ineq < -1e-6]).max())
            if (l_left_ineq > 1e-6).any():
                limits[1] = min(limits[1], (l_right_ineq[l_left_ineq > 1e-6] / l_left_ineq[l_left_ineq > 1e-6]).min())

            limits = tuple(limits)

            log(f"Found a bound. {out_a=}, {out_b=}, {out_c=}, {limits=}")
            if abs(out_a) < 1e-8 and abs(out_b) < 1e-8:
                return Constant(out_c, limits)
            if abs(out_a) < 1e-8:
                return Line(out_b, out_c, limits)
            return Quadratic(out_a, out_b, out_c, limits)
        except NoSolution:
            return Error()

    out = []
    for idx, ((lbd_1, sol_lbd_1), (lbd_2, sol_lbd_2)) in enumerate(itertools.pairwise(zip(lbds, gt_points))):
        log(f"Computing bound n°{idx}/{len(lbds) - 1}")
        if sol_lbd_1 is None or sol_lbd_2 is None:
            out.append({
                "bound": Error(),
                "timing": 0.0
            })
        else:
            bound_start = time.time()
            bound = get_bound(lbd_1, *sol_lbd_1, lbd_2, *sol_lbd_2)
            bound_end = time.time()
            out.append({
                "bound": bound,
                "timing": bound_end - bound_start
            })

    return {
        "timing": time.time() - start,
        "bounds": out
    }

@multi_bound
@lower_bound
def bound_lagrangian_line(problem, lbds, do_log=False, separate=False):
    assert isinstance(problem, Problem_sparse)
    log = get_logger(do_log)

    start = time.time()

    for idx in range(len(lbds)-1):
        if lbds[idx] < 0 < lbds[idx + 1]:
            log("Adding 0 to the lambdas", idx, lbds[:idx+1], lbds[idx+1:])
            lbds = np.concatenate([lbds[:idx+1], [0.0],lbds[idx+1:]])
            break

    gt_points = []
    for idx, lbd in enumerate(lbds):
        log(f"Solving 'ground truth problem' n°{idx}/{len(lbds)}")
        gt_points.append(get_sol_and_alpha(problem, lbd))

    log("Done computing ground truth")

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    def gen_problem():
        api = solve_api("cplex")
        api.add_var(a_1_eq.shape[1])
        api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
        api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
        return api

    part1 = gen_problem()
    part2 = gen_problem() if separate else part1

    def get_bound(lbd_1, sol_low, alpha_low_eq, alpha_low_ineq, lbd_2, sol_high, alpha_high_eq, alpha_high_ineq):
        if lbd_1 <= 0 and lbd_2 <= 0:
            mode = "minimize" if not problem.minimize else "maximize"
        else:
            mode = "minimize" if problem.minimize else "maximize"

        def const_coef():
            model = part1
            model.set_obj(c.transpose() + (-alpha_low_ineq.transpose() @ a_2_ineq) + (-alpha_low_eq.transpose() @ a_2_eq),
                          "minimize" if mini else "maximize")
            model.optimize()
            return model.get_objective() - (-alpha_low_eq.transpose() @ b_2_eq).flatten()[0] - (-alpha_low_ineq.transpose() @ b_2_ineq).flatten()[0]

        def linear_coef():
            model = part2
            model.set_obj(-alpha_low_eq.transpose() @ d_eq - alpha_low_ineq.transpose() @ d_ineq, mode)

            model.optimize()
            return model.get_objective()

        limits = [-np.inf, 0.0] if lbd_1 <= lbd_2 <= 0 else [0.0, np.inf]

        try:
            out_b = const_coef()
            out_a = linear_coef()

            limits = tuple(limits)

            log(f"Found a bound. {out_a=}, {out_b=}, {limits=}")
            if abs(out_a) < 1e-8 and abs(out_b) < 1e-8:
                return Constant(out_b, limits)
            return Line(out_a, out_b, limits)
        except NoSolution:
            return Error()

    out = []
    for idx, ((lbd_1, sol_lbd_1), (lbd_2, sol_lbd_2)) in enumerate(itertools.pairwise(zip(lbds, gt_points))):
        log(f"Computing bound n°{idx}/{len(lbds) - 1}")
        if sol_lbd_1 is None or sol_lbd_2 is None:
            out.append({
                "bound": Error(),
                "timing": 0.0
            })
        else:
            bound_start = time.time()
            bound = get_bound(lbd_1, *sol_lbd_1, lbd_2, *sol_lbd_2)
            bound_end = time.time()
            out.append({
                "bound": bound,
                "timing": bound_end - bound_start
            })

    return {
        "timing": time.time() - start,
        "bounds": out
    }