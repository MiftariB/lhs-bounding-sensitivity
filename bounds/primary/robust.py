import itertools
import signal
import time

from solveapi import solve_api
from bounds.bound_utils import Constant, Line, Minimize, Maximize, get_logger, Error
from bounds.primary import upper_bound, single_bound, multi_bound
from problems import Problem_sparse, solve_sparse, solve
import scipy.sparse as sp
import numpy as np


def matrix_view_problem(problem, lbd_1, lbd_2, lbd_opt):
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem
    _, nb_var = a_1_ineq.shape

    mid = (lbd_1 + lbd_2) / 2
    square = lbd_1 * lbd_2

    empty_csr_a_1 = sp.csr_array(([], ([], [])), shape=(a_1_eq.shape[0], nb_var))
    first_line_eq = sp.hstack([a_1_eq, empty_csr_a_1], format="csr")

    second_line_eq = sp.hstack([empty_csr_a_1, a_1_eq], format="csr")

    empty_csr_a_2 = sp.csr_array(([], ([], [])), shape=(a_2_eq.shape[0], nb_var))
    third_line_eq = sp.hstack([a_2_eq, empty_csr_a_2], format="csr")

    fourth_line_eq = sp.hstack([d_eq, a_2_eq], format="csr")

    fifth_line_eq = sp.hstack([empty_csr_a_2, d_eq], format="csr")

    first_line_ineq = sp.hstack([a_1_ineq, lbd_1 * a_1_ineq], format="csr")

    second_line_ineq = sp.hstack([a_1_ineq, lbd_2 * a_1_ineq], format="csr")

    third_line_ineq = sp.hstack([a_2_ineq + lbd_1 * d_ineq, lbd_1 * a_2_ineq + lbd_1 * lbd_1 * d_ineq], format="csr")

    fourth_line_ineq = sp.hstack([a_2_ineq + lbd_2 * d_ineq, lbd_2 * a_2_ineq + lbd_2 * lbd_2 * d_ineq], format="csr")

    a_2_plus_md = a_2_ineq + mid * d_ineq
    m_a_2_plus_dp = mid * a_2_ineq + square * d_ineq
    fifth_line_ineq = sp.hstack([a_2_plus_md, m_a_2_plus_dp], format="csr")

    full_matrix_eq = sp.vstack([first_line_eq, second_line_eq, third_line_eq, fourth_line_eq, fifth_line_eq],
                               format="csr")
    full_matrix_ineq = sp.vstack([first_line_ineq, second_line_ineq, third_line_ineq, fourth_line_ineq, fifth_line_ineq],
                                 format="csr")

    b_eq = np.concatenate([b_1_eq, np.zeros_like(b_1_eq), b_2_eq, np.zeros_like(b_2_eq), np.zeros_like(b_2_eq)])

    b_ineq = np.concatenate([b_1_ineq, b_1_ineq, b_2_ineq, b_2_ineq, b_2_ineq])

    return full_matrix_eq, b_eq, full_matrix_ineq, b_ineq

@single_bound
@upper_bound
def robust_concave_envelope(problem, lbd_1, lbd_2):
    assert isinstance(problem, Problem_sparse)
    c, mini = problem.c, problem.minimize
    a_1_ineq = problem.a_1_ineq

    lbd_opt, last_bnd = lbd_1, lbd_2
    lbd_opt += max(1e-5, (lbd_2-lbd_1)/100)

    full_matrix_eq, b_eq, full_matrix_ineq, b_ineq = matrix_view_problem(problem, lbd_1, lbd_2, lbd_opt)
    api = solve_api("cplex")
    api.activate_crossover()
    nb_var = a_1_ineq.shape[1]

    api.add_var(2 * nb_var)
    api.add_constr(full_matrix_eq, '==', b_eq.reshape(-1))
    api.add_constr(full_matrix_ineq, '<=', b_ineq.reshape(-1))
    all_lines = []
    full_objective_without_lambda = np.concatenate([c.transpose(), c.transpose()], axis=1)

    while (lbd_opt <= last_bnd):
        c_obj = np.concatenate([c.transpose(), lbd_opt * c.transpose()], axis=1)
        api.set_obj(c_obj.astype(float), "minimize" if mini else "maximize")

        api.optimize()
        status = api.get_status()
        if status == "unknown":
            break
        var_basis, constr_basis = api.get_basis()
        free_var = api.get_free()

        # Cutting each basis
        y_basis = np.zeros_like(var_basis)
        z_basis = np.zeros_like(var_basis)

        # Cutting to contrary
        z_not_basis = np.zeros_like(var_basis)
        y_non_basis = np.zeros_like(var_basis)

        # Assigning each value
        y_basis[0:nb_var] = var_basis[0:nb_var]
        y_non_basis[0:nb_var] = np.logical_not(var_basis[0:nb_var])
        z_not_basis[nb_var:2 * nb_var] = np.logical_not(var_basis[nb_var:2 * nb_var])
        z_basis[nb_var:2 * nb_var] = var_basis[nb_var:2 * nb_var]
        slack_basis = constr_basis
        slack_non_basis = np.logical_not(constr_basis)

        # Getting the size
        nb_basis_y = np.sum(y_basis)
        nb_non_basis_y = nb_var - nb_basis_y
        nb_basis_z = np.sum(z_basis)
        nb_non_basis_z = nb_var - nb_basis_z
        nb_basis_slack = np.sum(slack_basis)
        nb_non_basis_slack = len(slack_basis) - nb_basis_slack

        all_basis = np.concatenate([var_basis, constr_basis])
        c_b_z = np.zeros(api.model.linear_constraints.get_num(), dtype=float)
        c_b_z[nb_basis_y:nb_basis_y + nb_basis_z] = full_objective_without_lambda[0, z_basis]

        c_b_z_base = np.array(api.model.solution.advanced.btran(c_b_z))

        slacked_matrix_eq = sp.hstack([full_matrix_eq,
                                       0 * sp.eye(full_matrix_eq.shape[0], n=api.model.linear_constraints.get_num())],
                                       format="csr")

        slacked_matrix_ineq = sp.hstack([full_matrix_ineq, sp.eye(full_matrix_ineq.shape[0],
                                                                  n=api.model.linear_constraints.get_num(),
                                                                  k=slacked_matrix_eq.shape[0])], format="csr")
        big_M_matrix = sp.vstack([slacked_matrix_eq, slacked_matrix_ineq], format="csr")

        c_obj_full = np.zeros_like(all_basis, dtype=float)
        c_obj_full[0:2 * nb_var] = c_obj[0, 0:2 * nb_var]

        free_var_full = np.zeros_like(c_obj_full, dtype=bool)
        free_var_full[0:2 * nb_var] = free_var

        M_lhs = c_b_z_base @ big_M_matrix[:, (~all_basis)]

        M_rhs = np.zeros(nb_non_basis_y + nb_non_basis_z + nb_non_basis_slack)
        M_rhs[nb_non_basis_y:nb_non_basis_y + nb_non_basis_z] = c_obj[0, z_not_basis] / lbd_opt
        #M_rhs = M_rhs[~free_var]
        M_total = M_rhs - M_lhs

        dual_var = np.array(api.model.solution.get_dual_values())
        #reduced_costs_non_basic = c_obj_full[~all_basis] - (dual_var @ big_M_matrix)[~all_basis]
        reduced_costs_non_basic = c_obj_full[(~all_basis)] - (dual_var @ big_M_matrix)[(~all_basis)]

        non_zeros_in_M_total = M_total[np.abs(M_total) > 1e-6]
        non_zeros_in_non_basic_rc = reduced_costs_non_basic[np.abs(M_total) > 1e-6]
        conditions = - np.divide(non_zeros_in_non_basic_rc, non_zeros_in_M_total)

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

        full_sol = api.get_solution()
        y = full_sol[: nb_var]
        z = full_sol[nb_var:]
        all_lines.append(Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2)))
        lbd_opt += max(upper_bnd, 0) + max(1e-5, (lbd_2-lbd_1)/100)
        print(Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2)), lbd_opt)

        if np.isinf(upper_bnd):
            break

    if len(all_lines) == 0:
        return Error()

    if mini:
        return_object = Minimize(all_lines, limits=(lbd_1, lbd_2))
    else:
        return_object = Maximize(all_lines, limits=(lbd_1, lbd_2))

    return return_object


@single_bound
@upper_bound
def bound_robust_flat(problem, lbd_1, lbd_2):
    """
    min  c^t x
    s.t. a_1_ineq x <= b_1_ineq
         (a_2_ineq + lbd_1 d_ineq)x <= b_2_ineq
         (a_2_ineq + lbd_2 d_ineq)x <= b_2_ineq
         a_1_eq x = b_1_eq
         a_2_eq x = b_2_eq
         d_eq x = 0
    """
    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1])
    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    api.add_constr((a_2_ineq + lbd_1 * d_ineq), "<=", b_2_ineq.reshape(-1, ))
    api.add_constr((a_2_ineq + lbd_2 * d_ineq), "<=", b_2_ineq.reshape(-1, ))
    api.add_constr(a_2_eq, "==", b_2_eq.reshape(-1, ))
    api.add_constr(d_eq, "==", np.zeros((d_eq.shape[0])))
    api.set_obj(c.transpose(), "minimize" if mini else "maximize")
    api.optimize()
    status = api.get_status()

    if status != "unknown":
        objective = api.get_objective()
        return Constant(objective, (lbd_1, lbd_2))
    else:
        return None


@upper_bound
def bound_robust_line(problem, lbd_1, lbd_2, opti_first=None, opti_second=False):
    """
    min  c^t (y + lbd_1 z)
    s.t. a_1_ineq y + lbd_1 a_1_ineq z <= b_1_ineq
         a_1_ineq y + lbd_2 a_1_ineq z <= b_1_ineq
         (a_2_ineq + lbd_1 d_ineq)(y+lbd_1 z) <= b_2_ineq
         (a_2_ineq + lbd_2 d_ineq)(y+lbd_2 z) <= b_2_ineq
         a_2_ineq y + d_ineq z lbd_1 lbd_2 + (d_ineq y+a_2_ineq z)(lbd_1+lbd_2)/2 <= b_2_ineq

         a_1_eq y = b_1_eq
         a_1_eq z = 0
         a_2_eq y = b_2_eq
         a_2_eq z + d_eq y = 0
         d_eq z = 0
    """

    assert isinstance(problem, Problem_sparse)

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(2*nb_var)
    mid = (lbd_1 + lbd_2) / 2
    square = lbd_1 * lbd_2

    empty_csr_a_1 = sp.csr_array(([], ([], [])), shape=(a_1_eq.shape[0], nb_var))
    a_1_eq_y = sp.hstack([a_1_eq, empty_csr_a_1], format="csr")
    api.add_constr(a_1_eq_y, "==", b_1_eq.reshape(-1, ))

    a_1_eq_z = sp.hstack([empty_csr_a_1, a_1_eq], format="csr")
    api.add_constr(a_1_eq_z, "==", np.zeros((a_1_eq.shape[0])))

    empty_csr_a_2 = sp.csr_array(([], ([], [])), shape=(a_2_eq.shape[0], nb_var))
    a_2_eq_y = sp.hstack([a_2_eq, empty_csr_a_2], format="csr")
    api.add_constr(a_2_eq_y, "==", b_2_eq.reshape(-1, ))
    api.add_constr(sp.hstack([d_eq, a_2_eq], format="csr"), "==", np.zeros((a_2_eq.shape[0])))

    d_eq_z = sp.hstack([empty_csr_a_2, d_eq], format="csr")
    api.add_constr(d_eq_z, "==", np.zeros((d_eq.shape[0])))

    a_2_plus_md = a_2_ineq+mid*d_ineq
    m_a_2_plus_dp = mid*a_2_ineq + square*d_ineq
    sum_for_all_var = sp.hstack([a_2_plus_md, m_a_2_plus_dp], format="csr")

    api.add_constr(sum_for_all_var, "<=",  b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_2_ineq+lbd_1 * d_ineq, lbd_1*a_2_ineq + lbd_1*lbd_1 * d_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_2_ineq + lbd_2 * d_ineq, lbd_2 * a_2_ineq + lbd_2 * lbd_2 * d_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_1_ineq, lbd_1*a_1_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_1_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_1_ineq, lbd_2 * a_1_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_1_ineq.reshape(-1, ))

    if opti_first is None:
        opti_first = lbd_1

    api.set_obj(np.concatenate([c.transpose(), opti_first*c.transpose()], axis=1), "minimize" if mini else "maximize")
    api.optimize()
    status = api.get_status()
    if status == "unknown":
        #print(model.status)
        return None

    if opti_second is not False:
        if opti_second is None:
            opti_second = lbd_2
        first_obj = np.array([api.get_objective()])
        added_vec = np.concatenate([c.transpose(), opti_first*c.transpose()], axis=1)
        added_vec = sp.csr_array(added_vec)

        api.add_constr(added_vec, "==", first_obj)
        api.set_obj(np.concatenate([c.transpose(), opti_second*c.transpose()], axis=1),
                    "minimize" if mini else "maximize")
        api.optimize()

        if api.get_status() == "unknown":
            return None

    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z = full_sol[nb_var:]
    return Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2))

@multi_bound
@upper_bound
def bound_robust_fixed_slope_pairwise(problem, lbds, do_log=False):
    assert isinstance(problem, Problem_sparse)
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem

    log = get_logger(do_log)

    start = time.time()

    gt_points = []
    for idx, lbd in enumerate(lbds):
        log(f"Solving 'ground truth problem' n°{idx}/{len(lbds)}")
        gt_points.append(solve(problem, lbd))

    log("Done computing ground truth")

    out = []
    for idx, ((lbd_1, sol_1), (lbd_2, sol_2)) in enumerate(itertools.pairwise(zip(lbds, gt_points))):
        log(f"Computing bound n°{idx}/{len(lbds) - 1}")
        bound_start_time = time.time()
        try:
            bound = bound_robust_fixed_slope(problem, lbd_1, lbd_2, first_obj=sol_1, second_obj=sol_2)
        except:
            bound = Error()
        bound_end_time = time.time()
        out.append({"timing": bound_end_time - bound_start_time, "bound": bound})
    return {
        "timing": time.time() - start,
        "bounds": out
    }

@upper_bound
def bound_robust_fixed_slope(problem, lbd_1, lbd_2, first_obj=None, second_obj=None, fixed_slope=None):
    assert isinstance(problem, Problem_sparse)
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem
    if first_obj is None and fixed_slope is None:
        first_obj = solve_sparse(problem, lbd_1, debug=True)
    if second_obj is None and fixed_slope is None:
        second_obj = solve_sparse(problem, lbd_2, debug=True)

    if fixed_slope is None:
        slope = (second_obj-first_obj)/(lbd_2-lbd_1)
    else:
        slope = fixed_slope

    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(2 * nb_var)
    mid = (lbd_1 + lbd_2) / 2
    square = lbd_1 * lbd_2

    empty_csr_a_1 = sp.csr_array(([], ([], [])), shape=(a_1_eq.shape[0], nb_var))
    a_1_eq_y = sp.hstack([a_1_eq, empty_csr_a_1], format="csr")
    api.add_constr(a_1_eq_y, "==", b_1_eq.reshape(-1, ))

    a_1_eq_z = sp.hstack([empty_csr_a_1, a_1_eq], format="csr")
    api.add_constr(a_1_eq_z, "==", np.zeros((a_1_eq.shape[0])))

    empty_csr_a_2 = sp.csr_array(([], ([], [])), shape=(a_2_eq.shape[0], nb_var))
    a_2_eq_y = sp.hstack([a_2_eq, empty_csr_a_2], format="csr")
    api.add_constr(a_2_eq_y, "==", b_2_eq.reshape(-1, ))
    api.add_constr(sp.hstack([d_eq, a_2_eq], format="csr"), "==", np.zeros((a_2_eq.shape[0])))

    d_eq_z = sp.hstack([empty_csr_a_2, d_eq], format="csr")
    api.add_constr(d_eq_z, "==", np.zeros((d_eq.shape[0])))

    a_2_plus_md = a_2_ineq + mid * d_ineq
    m_a_2_plus_dp = mid * a_2_ineq + square * d_ineq
    sum_for_all_var = sp.hstack([a_2_plus_md, m_a_2_plus_dp], format="csr")

    api.add_constr(sum_for_all_var, "<=", b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_2_ineq + lbd_1 * d_ineq, lbd_1 * a_2_ineq + lbd_1 * lbd_1 * d_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_2_ineq + lbd_2 * d_ineq, lbd_2 * a_2_ineq + lbd_2 * lbd_2 * d_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_2_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_1_ineq, lbd_1 * a_1_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_1_ineq.reshape(-1, ))

    sum_for_all_var = sp.hstack([a_1_ineq, lbd_2 * a_1_ineq], format="csr")
    api.add_constr(sum_for_all_var, "<=", b_1_ineq.reshape(-1, ))

    fixed_slope = sp.csr_array(np.concatenate([np.zeros(c.transpose().shape),  c.transpose()], axis=1))
    api.add_constr(fixed_slope, "==", np.array([slope]))

    api.set_obj(np.concatenate([c.transpose(), np.zeros(c.transpose().shape)], axis=1),
                "minimize" if mini else "maximize")
    api.optimize()

    if api.get_status() == "unknown":
        return None
    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z = full_sol[nb_var:]
    return Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2))


@single_bound
@upper_bound
def bound_robust_xyflat(problem, lbd_1, lbd_2):
    flat_line = bound_robust_fixed_slope(problem, lbd_1, lbd_2, fixed_slope=0)
    if flat_line is None:
        return None
    return Constant(flat_line.b, (lbd_1, lbd_2))


@single_bound
@upper_bound
def bound_robust_line_left(problem, lbd_1, lbd_2):
    return bound_robust_line(problem, lbd_1, lbd_2, lbd_1, lbd_2)


@single_bound
@upper_bound
def bound_robust_line_right(problem, lbd_1, lbd_2):
    return bound_robust_line(problem, lbd_1, lbd_2, lbd_2, lbd_1)