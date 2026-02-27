import itertools
import time
import numpy as np
import scipy.sparse as sp

from bounds.bound_utils import Error, Constant, Line, Quadratic, Maximize, get_logger, Minimize
from bounds.primary import lower_bound, multi_bound
from problems import Problem_sparse, solve
from solveapi import solve_api, NoSolution


def get_sol_and_alpha(problem, lbd, coef_lbds=None):
    """ Solve the lagrangian (linearized) problem for a given lambda and return the solution and the dual variables 
    """
    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem

    api = solve_api("cplex")

    lb = [None if vt != 'B' else 0 for vt in var_types] if var_types is not None else None
    ub = [None if vt != 'B' else 1 for vt in var_types] if var_types is not None else None

    # here we ignore var_types to get the dual
    api.add_var(a_1_eq.shape[1], lb=lb, ub=ub)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    api.add_constr((a_2_eq + lbd * d_eq), "==", b_2_eq.reshape(-1, ))
    api.add_constr((a_2_ineq + lbd * d_ineq), "<=", b_2_ineq.reshape(-1, ))

    if coef_lbds is not None:
        lbd_1, lbd_2 = coef_lbds

        def change_d_matrix(array, lbd_1, lbd_2):
            return np.where(array > 0, lbd_1 * array, lbd_2 * array)
    
        new_d_matrix_data_ineq = change_d_matrix(d_ineq.data, lbd_1, lbd_2)
        new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)
        new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data), lbd_1, lbd_2)
        new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data), lbd_1, lbd_2)
        new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
        new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

        api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
        api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
        api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))

    d_eqs = slice(a_1_eq.shape[0] + a_1_ineq.shape[0],
                         a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0])

    d_ineqs = slice(a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0],
                           a_1_eq.shape[0] + a_1_ineq.shape[0] + a_2_eq.shape[0] + a_2_ineq.shape[0])

    api.set_obj(c.transpose(), "minimize" if mini else "maximize")
    api.optimize()

    status = api.get_status()
    if status == "unknown":
        return None
    dual = api.get_dual()
    optimal_lb_obj = api.get_objective()

    dual_eqs = dual[d_eqs]
    dual_ineqs = dual[d_ineqs]

    return optimal_lb_obj, dual_eqs, dual_ineqs


def get_sol_for_alpha(problem, lbd, alpha_eq, alpha_ineq):
    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem
    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(a_1_eq.shape[1], lb=[None]*nb_var, ub=([None]*nb_var), types=var_types)
    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    c_eq = (alpha_eq.transpose() @ (a_2_eq + lbd * d_eq))
    c_ineq = (alpha_ineq.transpose() @ (a_2_ineq + lbd * d_ineq))
    api.set_obj(c.transpose() - c_eq - c_ineq, "minimize" if mini else "maximize")

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
def bound_lagrangian_iter(problem, lbds, do_log=False):
    assert len(lbds) == 2 # TODO re-enable this
    assert isinstance(problem, Problem_sparse)
    log = get_logger(do_log)
    lbd_1, lbd_2 = lbds
    start = time.time()

    gt_points = []
    for idx, lbd in enumerate(lbds):
        log(f"Solving 'ground truth problem' n°{idx}/{len(lbds)}")
        gt_points.append(get_sol_and_alpha(problem, lbd))

    log("Done computing ground truth")

    # instance
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ ,_ = problem

    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1])

    def change_d_matrix(array):
        return np.where(array > 0, lbd_1 * array, lbd_2 * array)


    new_d_matrix_data_ineq = change_d_matrix(d_ineq.data)
    new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data))
    new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data))
    # We need to create two different matrices for the equality constraints
    # new_d_matrix_eq_pos = d_eq
    # new_d_matrix_eq_neg = d_eq
    new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
    new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
    api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
    api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))

    def get_sol_for_alpha(lbd, alpha_eq, alpha_ineq, lpmethod=0):
        api.set_obj(c.transpose() - (alpha_eq.transpose() @ (a_2_eq + lbd * d_eq))
                    - (alpha_ineq.transpose() @ (a_2_ineq + lbd * d_ineq)),
                    "minimize" if mini else "maximize")

        api.optimize(lpmethod=lpmethod)
        status = api.get_status()
        if status == "unknown":
            return None
        return api.get_objective() + (alpha_eq.transpose() @ b_2_eq) + (alpha_ineq.transpose() @ b_2_ineq)

    sols_1 = []
    sols_2 = []
    sol_lbd_1 = gt_points[0]
    sol_lbd_2 = gt_points[1]
    x = np.linspace(lbd_1, lbd_2, 10)
    for j, i in enumerate(x):
        log(f"Solving left {j}/{len(x)}")
        if len(sols_1) and sols_1[-1] is None:
            sols_1.append(None)
            continue

        sols_1.append(get_sol_for_alpha(i, sol_lbd_1[1], sol_lbd_1[2], lpmethod=4 if j == 0 else 0))
    for j, i in enumerate(reversed(x)):
        log(f"Solving right {j}/{len(x)}")
        if len(sols_2) and sols_2[-1] is None:
            sols_2.append(None)
            continue

        sols_2.append(get_sol_for_alpha(i, sol_lbd_2[1], sol_lbd_2[2], lpmethod=4 if j == 0 and sols_1[-1] is None else 0))
    sols_2 = list(reversed(sols_2))

    return x, sols_1, sols_2


@multi_bound
@lower_bound
def bound_lagrangian_flat(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef=False, segment=False, do_log=do_log)

@multi_bound
@lower_bound
def bound_lagrangian_flat_coef(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef=True, segment=False, do_log=do_log)

@multi_bound
@lower_bound
def bound_lagrangian_flat_coef_adv(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef="adv", segment=False, do_log=do_log)

@multi_bound
@lower_bound
def bound_lagrangian_bisegment(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef=False, segment=True, do_log=do_log)

@multi_bound
@lower_bound
def bound_lagrangian_bisegment_coef(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef=True, segment=True, do_log=do_log)

@multi_bound
@lower_bound
def bound_lagrangian_bisegment_coef_adv(problem, lbds, do_log=False):
    return bound_lagrangian_flat_internal(problem, lbds, coef="adv", segment=True, do_log=do_log)

def bound_lagrangian_flat_internal(problem, lbds, coef, segment, do_log=False):
    assert isinstance(problem, Problem_sparse)

    if coef:
        problem = problem.positive()
    
    #if segment:
    #    # bi-segment only works on linear problems
    #    problem = problem.linear()

    log = get_logger(do_log)
    start = time.time()

    if coef != "adv":
        gt_points = []
        for idx, lbd in enumerate(lbds):
            log(f"Solving 'linear ground truth problem' n°{idx}/{len(lbds)}")
            gt_points.append(get_sol_and_alpha(problem, lbd))
    else:
        gt_points = None

    log("Done computing ground truth")

    # instance
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem

    is_milp = ('I' in var_types or 'B' in var_types) if var_types is not None else False
    
    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1], types=var_types)

    def change_d_matrix(array, lbd_1, lbd_2):
        return np.where(array > 0, lbd_1 * array, lbd_2 * array)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    
    def get_sol_for_alpha(lbd, alpha_eq, alpha_ineq):
        api.set_obj(c.transpose() - (alpha_eq.transpose() @ (a_2_eq + lbd * d_eq))
                    - (alpha_ineq.transpose() @ (a_2_ineq + lbd * d_ineq)),
                    "minimize" if mini else "maximize")

        api.optimize()
        status = api.get_status()
        if status == "unknown":
            out = None
        else:
            out = api.get_objective() + (alpha_eq.transpose() @ b_2_eq) + (alpha_ineq.transpose() @ b_2_ineq)

        return out

    def get_bound(lbd_1, sol_low, alpha_low_eq, alpha_low_ineq, lbd_2, sol_high, alpha_high_eq, alpha_high_ineq):
        if coef:
            new_d_matrix_data_ineq = change_d_matrix(d_ineq.data, lbd_1, lbd_2)
            new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)
            new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data), lbd_1, lbd_2)
            new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data), lbd_1, lbd_2)
            new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
            new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

            first_idx, _ = api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
            api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
            _, last_idx = api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))
        
        line2_l = get_sol_for_alpha(lbd_1, alpha_high_eq, alpha_high_ineq)
        line2_r = sol_high if coef is False and not is_milp else get_sol_for_alpha(lbd_2, alpha_high_eq, alpha_high_ineq)

        line1_l = sol_low if coef is False and not is_milp else get_sol_for_alpha(lbd_1, alpha_low_eq, alpha_low_ineq)
        line1_r = get_sol_for_alpha(lbd_2, alpha_low_eq, alpha_low_ineq)

        if coef:
            api.delete_constr(list(range(first_idx, last_idx)))
        
        if line1_r is not None and line1_l is not None:
            if segment:
                slope1 = (line1_r - line1_l)/(lbd_2-lbd_1)
                line1 = Line(slope1, line1_l - slope1 * lbd_1, (lbd_1, lbd_2))
            elif problem.minimize:
                line1 = Constant(min(line1_l, line1_r), (lbd_1, lbd_2))
            else:
                line1 = Constant(max(line1_l, line1_r), (lbd_1, lbd_2))
        else:
            line1 = Error()

        if line2_l is not None and line2_r is not None:
            if segment:
                slope2 = (line2_r - line2_l)/(lbd_2-lbd_1)
                line2 = Line(slope2, line2_r - slope2*lbd_2, (lbd_1, lbd_2))
            elif problem.minimize:
                line2 = Constant(min(line2_l, line2_r), (lbd_1, lbd_2))
            else:
                line2 = Constant(max(line2_l, line2_r), (lbd_1, lbd_2))
        else:
            line2 = Error()
        
        if isinstance(line1, Error) and isinstance(line2, Error):
            return Error()
        elif isinstance(line1, Error):
            return line2
        elif isinstance(line2, Error):
            return line1
        else:
            if problem.minimize:
                return Maximize([line1, line2], limits=(lbd_1, lbd_2))
            else:
                return Minimize([line1, line2], limits=(lbd_1, lbd_2))

    out = []
    for cnt, (idx_1, idx_2) in enumerate(itertools.pairwise(range(len(lbds)))):
        lbd_1 = lbds[idx_1]
        lbd_2 = lbds[idx_2]
        log(f"Computing bound n°{cnt}/{len(lbds)-1}")
        if coef != "adv":
            sol_lbd_1 = gt_points[idx_1]
            sol_lbd_2 = gt_points[idx_2]
        else:
            sol_lbd_1 = get_sol_and_alpha(problem, lbd_1, (lbd_1, lbd_2))
            sol_lbd_2 = get_sol_and_alpha(problem, lbd_2, (lbd_1, lbd_2))

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
    lbd_1, lbd_2 = min(lbds), max(lbds)
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

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _, _ = problem

    def change_d_matrix(array):
        return np.where(array > 0, lbd_1 * array, lbd_2 * array)

    def gen_problem():
        api = solve_api("cplex")
        api.add_var(a_1_eq.shape[1])
        new_d_matrix_data_ineq = change_d_matrix(d_ineq.data)
        new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

        new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data))
        new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data))
        # We need to create two different matrices for the equality constraints
        # new_d_matrix_eq_pos = d_eq
        # new_d_matrix_eq_neg = d_eq
        new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
        new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

        api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
        api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
        api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
        api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
        api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))
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

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _, _ = problem

    def change_d_matrix(array):
        lbd_1, lbd_2 = min(lbds), max(lbds)
        return np.where(array > 0, lbd_1 * array, lbd_2 * array)

    def gen_problem():
        api = solve_api("cplex")
        api.add_var(a_1_eq.shape[1])
        new_d_matrix_data_ineq = change_d_matrix(d_ineq.data)
        new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

        new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data))
        new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data))
        # We need to create two different matrices for the equality constraints
        # new_d_matrix_eq_pos = d_eq
        # new_d_matrix_eq_neg = d_eq
        new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
        new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

        api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
        api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
        api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
        api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
        api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))
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
