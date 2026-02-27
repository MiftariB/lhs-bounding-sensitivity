from solveapi import solve_api
from bounds.bound_utils import Constant, Line
from bounds.primary import bi_bound, upper_bound, single_bound
from problems import Problem_sparse
import scipy.sparse as sp
import numpy as np

def coefficient_flat_upper(problem, lbd_1, lbd_2):
    """
        min  c^t x
        s.t. a_1_ineq x <= b_1_ineq
             sum(a_2_ineq_i + ldb_1 d_ineq_i)x <= b_2_ineq_i if d_ineq_i <= 0
             sum(a_2_ineq_i + ldb_2 d_ineq_i)x <= b_2_ineq_i if d_ineq_i > 0
             a_1_eq x = b_1_eq
             a_2_eq x = b_2_eq
             d_eq x = 0
    """
    assert lbd_1 < lbd_2, "lbd_1 must be less than lbd_2"
    assert isinstance(problem, Problem_sparse)
    problem = problem.positive()
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem.positive()

    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1], types=var_types)
    def change_d_matrix(array):
        return np.where(array < 0, lbd_1 * array, lbd_2 * array)

    new_d_matrix_data = change_d_matrix(d_ineq.data)
    new_d_matrix = sp.csr_matrix((new_d_matrix_data, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    api.add_constr((a_2_ineq + new_d_matrix), "<=", b_2_ineq.reshape(-1, ))
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

def coefficient_flat_lower(problem, lbd_1, lbd_2):
    """
        min  c^t x
        s.t. a_1_ineq x <= b_1_ineq
             sum(a_2_ineq_i + ldb_1 d_ineq_i)x <= b_2_ineq_i if d_ineq_i > 0
             sum(a_2_ineq_i + ldb_2 d_ineq_i)x <= b_2_ineq_i if d_ineq_i < 0
    """

    assert isinstance(problem, Problem_sparse)
    problem = problem.positive()
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem

    api = solve_api("cplex")
    api.add_var(a_1_eq.shape[1], types=var_types)
    def change_d_matrix(array):
        return np.where(array > 0, lbd_1 * array, lbd_2 * array)
    #print(d_ineq.data)

    new_d_matrix_data_ineq = change_d_matrix(d_ineq.data)
    #print(new_d_matrix_data_ineq)
    new_d_matrix_ineq = sp.csr_matrix((new_d_matrix_data_ineq, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    new_d_matrix_data_eq_pos = change_d_matrix(np.array(d_eq.data))
    new_d_matrix_data_eq_neg = change_d_matrix(-np.array(d_eq.data))
    # We need to create two different matrices for the equality constraints
    #new_d_matrix_eq_pos = d_eq
    #new_d_matrix_eq_neg = d_eq
    new_d_matrix_eq_pos = sp.csr_matrix((new_d_matrix_data_eq_pos, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
    new_d_matrix_eq_neg = sp.csr_matrix((new_d_matrix_data_eq_neg, d_eq.indices, d_eq.indptr), shape=d_eq.shape)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))
    api.add_constr((a_2_ineq + new_d_matrix_ineq), "<=", b_2_ineq.reshape(-1, ))
    api.add_constr(a_2_eq + new_d_matrix_eq_pos, "<=", b_2_eq.reshape(-1, ))
    api.add_constr(-a_2_eq + new_d_matrix_eq_neg, "<=", -b_2_eq.reshape(-1, ))
    #api.add_constr(sp.hstack([sp.identity(a_1_eq.shape[1]/2), -sp.identity(a_1_eq.shape[1]/2)]), ">=", np.zeros((int(a_1_eq.shape[1]/2),)))
    #api.add_constr(sp.identity(a_1_eq.shape[1]), ">=", np.zeros((a_1_eq.shape[1],)))

    api.set_obj(c.transpose(), "minimize" if mini else "maximize")

    #print(rstat)
    #exit(0)

    api.optimize()
    status = api.get_status()
    if status != "unknown":
        objective = api.get_objective()
        return Constant(objective, (lbd_1, lbd_2))
    else:
        return None

@single_bound
@bi_bound(coefficient_flat_lower, coefficient_flat_upper)
def coefficient_flat(problem, lbd_1, lbd_2):
    return coefficient_flat_lower(problem, lbd_1, lbd_2)

#@single_bound
@upper_bound
def bound_coefficient_line(problem, lbd_1, lbd_2, opti_first=None, opti_second=False):
    """
    min  c^t (y + lbd_1 z)
    a_1_ineq (y+ lbd z) <= b_1_ineq
    a_1_eq (y+lbd z) = b_1_eq

    a_2 (y+lbd z) + lbd d (y+lbd z) <= b
    a_2 y+ lbd a_2 z + lbd d y + lbd^2 d z <= b
    max(lbd d y) + max(a_2 y+ lbd a_2 z + lbd^2 d z) <= b
    """
    print(opti_first, opti_second)
    assert isinstance(problem, Problem_sparse)
    assert lbd_1 < lbd_2

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _, _ = problem

    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(3*nb_var, lb=[0]*3*nb_var, ub=([None]*3*nb_var))

    def max_lbd_affine(array):
        return np.maximum.reduce([lbd_1 * array, lbd_2 * array])

    def max_lbd_square(array):
        if lbd_1 < 0 < lbd_2:
            return np.maximum.reduce([lbd_1**2 * array, lbd_2**2 * array, np.zeros_like(array)])
        else:
            return np.maximum.reduce([lbd_1**2 * array, lbd_2**2 * array])
    #print(d_ineq.data)

    #a_1_ineq y+ lbd a_1_ineq z1 - lbd a_1_ineq z2  <= b_1_ineq

    a_1_ineq_z_1_data = max_lbd_affine(a_1_ineq.data)
    a_1_ineq_z_1 = sp.csr_matrix((a_1_ineq_z_1_data, a_1_ineq.indices, a_1_ineq.indptr), shape=a_1_ineq.shape)
    a_1_ineq_z_2_data = max_lbd_affine(-a_1_ineq.data)
    a_1_ineq_z_2 = sp.csr_matrix((a_1_ineq_z_2_data, a_1_ineq.indices, a_1_ineq.indptr), shape=a_1_ineq.shape)

    a_1_ineq_csr = sp.hstack([a_1_ineq, a_1_ineq_z_1, a_1_ineq_z_2], format="csr")
    api.add_constr(a_1_ineq_csr, "<=", b_1_ineq.reshape(-1, ))


    empty_matrix = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(a_1_eq.shape[0], a_1_eq.shape[1]))
    a_1_eq_csr = sp.hstack([a_1_eq, empty_matrix, empty_matrix], format="csr")
    api.add_constr(a_1_eq_csr, "==", b_1_eq.reshape(-1, ))

    a_1_eq_z_csr = sp.hstack([empty_matrix, a_1_eq, -a_1_eq], format="csr")
    api.add_constr(a_1_eq_z_csr, "==", np.zeros((a_1_eq.shape[0],)))

    # a_2_ineq y + lbd a_2_ineq z1 - lbd a_2_ineq z2 + lbd d_ineq y + lbd^2 d_ineq z1 - lbd^2 d_ineq z2 <= b_2_ineq
    a_2_ineq_z_1_data = max_lbd_affine(a_2_ineq.data)

    a_2_ineq_z_1 = sp.csr_matrix((a_2_ineq_z_1_data, a_2_ineq.indices, a_2_ineq.indptr), shape=a_2_ineq.shape)
    a_2_ineq_z_2_data = max_lbd_affine(-a_2_ineq.data)
    a_2_ineq_z_2 = sp.csr_matrix((a_2_ineq_z_2_data, a_2_ineq.indices, a_2_ineq.indptr), shape=a_2_ineq.shape)

    d_ineq_z_1_data = max_lbd_square(d_ineq.data)
    d_ineq_z_1 = sp.csr_matrix((d_ineq_z_1_data, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)
    d_ineq_z_2_data = max_lbd_square(-d_ineq.data)
    d_ineq_z_2 = sp.csr_matrix((d_ineq_z_2_data, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    d_ineq_y_data = max_lbd_affine(d_ineq.data)
    d_ineq_y = sp.csr_matrix((d_ineq_y_data, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    result_y = a_2_ineq + d_ineq_y
    result_z_1 = a_2_ineq_z_1 + d_ineq_z_1
    result_z_2 = a_2_ineq_z_2 + d_ineq_z_2
    a_2_ineq_csr = sp.hstack([result_y, result_z_1, result_z_2], format="csr")
    api.add_constr(a_2_ineq_csr, "<=", b_2_ineq.reshape(-1, ))

    # a_2_eq y + lbd a_2_eq z1 - lbd a_2_eq z2 + lbd d_eq y + lbd^2 d_eq z1 - lbd^2 d_eq z2 = b_2_eq
    # Derived as
    # a_2_eq y == b_2_eq
    # d_eq y + a_2_eq z1 - a_2_eq z2 == 0
    # d_eq z1 - d_eq z2 == 0

    ## a_2_eq y == b_2_eq
    empty_csr_a_2 = sp.csr_array(([], ([], [])), shape=(a_2_eq.shape[0], nb_var))
    a_2_eq_y = sp.hstack([a_2_eq, empty_csr_a_2, empty_csr_a_2], format="csr")
    api.add_constr(a_2_eq_y, "==", b_2_eq.reshape(-1, ))

    ## d_eq y + a_2_eq z1 - a_2_eq z2 == 0
    second_eq_matrix = sp.hstack([d_eq, a_2_eq, -a_2_eq], format="csr")
    api.add_constr(second_eq_matrix, "==", np.zeros((d_eq.shape[0],)))

    ## d_eq z1 - d_eq z2 == 0
    third_eq_matrix = sp.hstack([empty_csr_a_2, d_eq, -d_eq], format="csr")
    api.add_constr(third_eq_matrix, "==", np.zeros((d_eq.shape[0],)))

    # the variables used must be positive
    identity = sp.eye(nb_var)
    pos_constr_1 = sp.hstack([identity, lbd_1*identity, -lbd_1*identity], format="csr")
    api.add_constr(pos_constr_1, ">=", np.zeros((nb_var,)))
    pos_constr_2 = sp.hstack([identity, lbd_2*identity, -lbd_2*identity], format="csr")
    api.add_constr(pos_constr_2, ">=", np.zeros((nb_var,)))

    if opti_first is None:
        opti_first = lbd_2

    api.set_obj(np.concatenate([c.transpose(), opti_first*c.transpose(), -opti_first*c.transpose()], axis=1),
                "minimize" if mini else "maximize")
    api.optimize()
    status = api.get_status()
    if status == "unknown":
        #print(model.status)
        return None

    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z1 = full_sol[nb_var:2 * nb_var]
    z2 = full_sol[2 * nb_var:]
    z = z1 - z2

    print(sum(c.transpose() @ y + c.transpose() @ z * lbd_1), sum(c.transpose() @ y + c.transpose() @ z * lbd_2))

    if opti_second is not False:
        if opti_second is None:
            opti_second = lbd_1
        first_obj = np.array([api.get_objective()])
        added_vec = np.concatenate([c.transpose(), opti_first * c.transpose(), -opti_first * c.transpose()], axis=1)
        added_vec = sp.csr_array(added_vec)
        api.add_constr(added_vec, "==", first_obj)
        api.set_obj(np.concatenate([c.transpose(), opti_second * c.transpose(), -opti_second * c.transpose()], axis=1),
                    "minimize" if mini else "maximize")
        api.optimize()

        if api.get_status() == "unknown":
            print("api.model.status:", api.model.solution.get_status())
            return None

    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z1 = full_sol[nb_var:2*nb_var]
    z2 = full_sol[2*nb_var:]
    z = z1 - z2
    print(z)
    print(sum(c.transpose() @ y + lbd_1*c.transpose() @z), sum(c.transpose() @ y + lbd_2* c.transpose() @z))

    return Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2))


@upper_bound
def combined_bound_coefficient_line(problem, lbd_1, lbd_2, opti_first=None, opti_second=False):
    """
    min  c^t (y + lbd_1 z)
    a_1_ineq (y+ lbd z) <= b_1_ineq
    a_1_eq (y+lbd z) = b_1_eq

    a_2 (y+lbd z) + lbd d (y+lbd z) <= b
    a_2 y+ lbd a_2 z + lbd d y + lbd^2 d z <= b
    max(lbd d y) + max(a_2 y+ lbd a_2 z + lbd^2 d z) <= b
    """

    assert isinstance(problem, Problem_sparse)
    assert lbd_1 < lbd_2

    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _, _ = problem

    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(3*nb_var, lb=[0]*3*nb_var, ub=([None]*3*nb_var))

    def max_lbd_affine(array):
        return np.where(array < 0, lbd_1 * array, lbd_2 * array)

    def max_lbd_square_two_matrix(squared_matrix, affine_matrix):

        lbd_1_matrix = lbd_1**2 * squared_matrix + lbd_1 * affine_matrix
        lbd_2_matrix = lbd_2**2 * squared_matrix + lbd_2 * affine_matrix
        max_lbd = np.maximum.reduce([lbd_1_matrix.data, lbd_2_matrix.data])
        max_lbd_matrix = sp.csr_matrix((max_lbd, lbd_2_matrix.indices, lbd_2_matrix.indptr),
                                       shape=lbd_2_matrix.shape)

        aff_coo = affine_matrix
        s_coo = squared_matrix.tocoo()
        s_coo_row = s_coo.row
        s_coo_col = s_coo.col
        s_coo = squared_matrix

        for i in range(len(s_coo_row)):
            r = s_coo_row[i]
            c = s_coo_col[i]
            mid_max = (-aff_coo[r, c])/(2*s_coo[r, c])
            if lbd_1 <= mid_max and mid_max <= lbd_2:

                total_term = mid_max**2*s_coo[r, c] + mid_max*aff_coo[r, c]
                print(r, c, total_term, max_lbd_matrix[r, c])

                max_lbd_matrix[r, c] = max(max_lbd_matrix[r, c],
                                           total_term)

        return max_lbd_matrix
    #print(d_ineq.data)

    #a_1_ineq y+ lbd a_1_ineq z1 - lbd a_1_ineq z2  <= b_1_ineq

    a_1_ineq_z_1_data = max_lbd_affine(a_1_ineq.data)
    a_1_ineq_z_1 = sp.csr_matrix((a_1_ineq_z_1_data, a_1_ineq.indices, a_1_ineq.indptr), shape=a_1_ineq.shape)
    a_1_ineq_z_2_data = max_lbd_affine(-a_1_ineq.data)
    a_1_ineq_z_2 = sp.csr_matrix((a_1_ineq_z_2_data, a_1_ineq.indices, a_1_ineq.indptr), shape=a_1_ineq.shape)

    a_1_ineq_csr = sp.hstack([a_1_ineq, a_1_ineq_z_1, a_1_ineq_z_2], format="csr")
    api.add_constr(a_1_ineq_csr, "<=", b_1_ineq.reshape(-1, ))

    # a_1_eq == b_1_eq
    empty_matrix = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(a_1_eq.shape[0], a_1_eq.shape[1]))
    a_1_eq_csr = sp.hstack([a_1_eq, empty_matrix, empty_matrix], format="csr")
    api.add_constr(a_1_eq_csr, "==", b_1_eq.reshape(-1, ))

    # a_1_eq z1 - a_1_eq z2 == 0
    a_1_eq_z_csr = sp.hstack([empty_matrix, a_1_eq, -a_1_eq], format="csr")
    api.add_constr(a_1_eq_z_csr, "==", np.zeros((a_1_eq.shape[0],)))

    # a_2_ineq y + lbd a_2_ineq z1 - lbd a_2_ineq z2 + lbd d_ineq y + lbd^2 d_ineq z1 - lbd^2 d_ineq z2 <= b_2_ineq

    d_ineq_y_data = max_lbd_affine(d_ineq.data)
    d_ineq_y = sp.csr_matrix((d_ineq_y_data, d_ineq.indices, d_ineq.indptr), shape=d_ineq.shape)

    result_z_1 = max_lbd_square_two_matrix(d_ineq, a_2_ineq)
    result_z_2 = max_lbd_square_two_matrix(-d_ineq, -a_2_ineq)

    result_y = a_2_ineq + d_ineq_y

    a_2_ineq_csr = sp.hstack([result_y, result_z_1, result_z_2], format="csr")
    api.add_constr(a_2_ineq_csr, "<=", b_2_ineq.reshape(-1, ))
    print("hi3")
    # a_2_eq y + lbd a_2_eq z1 - lbd a_2_eq z2 + lbd d_eq y + lbd^2 d_eq z1 - lbd^2 d_eq z2 = b_2_eq
    assert a_2_eq.shape[0] == 0, "coefficient bound does not support equality constraints for a_2_eq"


    """
    a_2_eq_z_1_data = max_lbd_affine(a_2_eq.data.copy())
    a_2_eq_z_1 = sp.csr_matrix((a_2_eq_z_1_data, a_2_eq.indices, a_2_eq.indptr), shape=a_2_eq.shape)
    a_2_eq_z_2_data = max_lbd_affine(-a_2_eq.data.copy())
    a_2_eq_z_2 = sp.csr_matrix((a_2_eq_z_2_data, a_2_eq.indices, a_2_eq.indptr), shape=a_2_eq.shape)
    d_eq_z_1_data = max_lbd_square(d_eq.data.copy())
    d_eq_z_1 = sp.csr_matrix((d_eq_z_1_data, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
    d_eq_z_2_data = max_lbd_square(-d_eq.data.copy())
    d_eq_z_2 = sp.csr_matrix((d_eq_z_2_data, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
    d_eq_y_data = max_lbd_affine(d_eq.data.copy())
    d_eq_y = sp.csr_matrix((d_eq_y_data, d_eq.indices, d_eq.indptr), shape=d_eq.shape)
    result_y = a_2_eq + d_eq_y
    result_z_1 = a_2_eq_z_1 + d_eq_z_1
    result_z_2 = a_2_eq_z_2 + d_eq_z_2
    a_2_eq_csr = sp.hstack([result_y, result_z_1, result_z_2], format="csr")
    api.add_constr(a_2_eq_csr, "==", b_2_eq.reshape(-1, ))
    """
    print("hi4")

    if opti_first is None:
        opti_first = lbd_2

    api.set_obj(np.concatenate([c.transpose(), opti_first*c.transpose(), -opti_first*c.transpose()], axis=1),
                "minimize" if mini else "maximize")
    api.optimize()
    status = api.get_status()
    if status == "unknown":
        #print(model.status)
        return None

    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z1 = full_sol[nb_var:2 * nb_var]
    z2 = full_sol[2 * nb_var:]
    z = z1 - z2

    print(sum(c.transpose() @ y + c.transpose() @ z * lbd_1), sum(c.transpose() @ y + c.transpose() @ z * lbd_2))

    if opti_second is not False:
        if opti_second is None:
            opti_second = lbd_1
        first_obj = np.array([api.get_objective()])
        added_vec = np.concatenate([c.transpose(), opti_first * c.transpose(), -opti_first * c.transpose()], axis=1)
        added_vec = sp.csr_array(added_vec)
        api.add_constr(added_vec, "==", first_obj)
        api.set_obj(np.concatenate([c.transpose(), opti_second * c.transpose(), -opti_second * c.transpose()], axis=1),
                    "minimize" if mini else "maximize")
        api.optimize()

        if api.get_status() == "unknown":
            print("api.model.status:", api.model.solution.get_status())
            return None

    full_sol = api.get_solution()
    y = full_sol[: nb_var]
    z1 = full_sol[nb_var:2*nb_var]
    z2 = full_sol[2*nb_var:]
    z = z1 - z2

    print(sum(c.transpose() @ y + c.transpose() @z* lbd_1), sum(c.transpose() @ y + c.transpose() @z* lbd_2))

    return Line(c.transpose() @ z, (c.transpose() @ y), (lbd_1, lbd_2))
