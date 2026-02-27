from bounds.bound_utils import Error
from bounds.primary import upper_bound
from problems import convert_to_eqproblem, solve
import numpy as np
import scipy

@upper_bound
def bound_basis_eigen(problem, lbd, _):
    eq_problem = convert_to_eqproblem(problem)
    _, basis = solve(eq_problem, lbd, return_basis=True)

    # Put the constraints in the form (A_b + lbd' D_b) x_b == [b_1, b_2]
    # with lbd'=0 around lbd.
    A = np.concatenate([eq_problem.a_1, eq_problem.a_2 + lbd * eq_problem.d])
    A_b = A[:, basis]
    D_b = np.concatenate([np.zeros((eq_problem.a_1.shape[0], len(basis))), eq_problem.d[:, basis]])

    c_b = eq_problem.c[basis, :]

    A_b_inv = np.linalg.inv(A_b)
    x_star = A_b_inv @ np.concatenate([eq_problem.b_1, eq_problem.b_2])

    # print(np.concatenate([eq_problem.b_1, eq_problem.b_2]))
    # print(A_b @ x_star)

    eig_val, Q = scipy.linalg.eig(A_b_inv @ D_b)

    if np.linalg.matrix_rank(Q) != len(basis):
        return Error()  # need full rank to inverse...

    L = np.diag(eig_val)

    try:
        Q_inv = np.linalg.inv(Q)
    except:
        return Error()

    # These two are the same, checked
    # print(Q @ L @ Q_inv)
    # print(A_b_inv @ D_b)
    # display(Q@Q_inv)
    # display(Q_inv@Q)
    # sns.heatmap(Q@Q_inv)

    # We will be inverting  the diagonal matrix (I + lbd_bis * L) soon. Let's find values of lbd_bis where there is a 0 on the diagonal
    min_lbd_bis, max_lbd_bis = eq_problem.range
    min_lbd_bis = np.nextafter(min_lbd_bis - lbd, min_lbd_bis - lbd - 1)
    max_lbd_bis = np.nextafter(max_lbd_bis - lbd, max_lbd_bis - lbd + 1)

    for i in range(len(basis)):
        if abs(eig_val[i]) > 1e-9:
            v = -1 / eig_val[i]
            if v <= 0:
                min_lbd_bis = max(min_lbd_bis, v)
            if v >= 0:
                max_lbd_bis = min(max_lbd_bis, v)

    min_lbd_bis += lbd
    max_lbd_bis += lbd

    def evaluate(space):
        valid_limit = (space > min_lbd_bis) & (space < max_lbd_bis)

        if valid_limit.sum() == 0:
            return None

        lbd_bis = space[valid_limit].reshape((-1, 1, 1))
        # lbd_bis = lbd
        inv_diag = np.linalg.inv(np.identity(len(basis)) + (lbd_bis - lbd) * L)
        x = Q @ inv_diag @ Q_inv @ x_star
        # print("x_star", x_star)
        # print("x", x)
        assert abs(np.imag(x).max()) <= 1e-6 and abs(np.imag(x).min()) <= 1e-6
        x = np.real(x)

        # print("out", ((A_b + (lbd_bis - lbd) * D_b) @ x))
        # print("b", np.concatenate([eq_problem.b_1, eq_problem.b_2]))
        # TODO: it's cheating. we are drawing line between values but we are not sure they are valid
        valid = (x < 0).sum(axis=(1, 2)) == 0
        if valid.sum() == 0:
            return None
        lbd_bis = lbd_bis[valid, :, :]
        x = x[valid, :, :]
        # print(x.min())

        obj = c_b.transpose() @ x

        out = space.copy()
        out[:] = float("nan")
        out[np.indices(space.shape)[0, :][valid_limit][valid]] = obj[:, 0, 0]

        return out

    return evaluate


@upper_bound
def bound_basis_schur(problem, lbd, _):
    eq_problem = convert_to_eqproblem(problem)

    _, basis = solve(eq_problem, lbd, return_basis=True)

    # Put the constraints in the form (A_b + lbd' D_b) x_b == [b_1, b_2]
    # with lbd'=0 around lbd.

    A = np.concatenate([eq_problem.a_1, eq_problem.a_2 + lbd * eq_problem.d])
    A_b = A[:, basis]
    D_b = np.concatenate([np.zeros((eq_problem.a_1.shape[0], len(basis))), eq_problem.d[:, basis]])
    A_b_inv = np.linalg.inv(A_b)
    c_b = eq_problem.c[basis, :]

    E = A_b_inv @ D_b
    x_star = A_b_inv @ np.concatenate([eq_problem.b_1, eq_problem.b_2])

    T, Z = scipy.linalg.schur(E, 'complex')

    if np.diag(T).max() > 1e-9:
        lambda_min = -1 / np.diag(T).max()
    else:
        lambda_min = -np.inf

    if np.diag(T).min() < -1e-9:
        lambda_max = -1 / np.diag(T).min()
    else:
        lambda_max = np.inf
    # (I + lbd*Z T Z^H) x = x^*
    # Z (I + lbd T) Z^H x = x^*
    # x = Z (I + lbd T)^-1 Z^H x^*
    # c^t x = c^t Z (I + lbd T)^-1 Z^H x^*
    left = Z
    right = Z.conj().T @ x_star

    def evaluate(space):
        valid_limit = (space > lambda_min + lbd) & (space < lambda_max + lbd)

        if valid_limit.sum() == 0:
            return None

        x = np.array([np.real(left @ scipy.linalg.solve_triangular(np.identity(len(basis)) + (l - lbd) * T,
                                                                   np.identity(len(basis)) @ right)) for l in
                      space[valid_limit]])

        mask = x.min(axis=(1, 2)) < -1e-9
        if mask.sum() == mask.shape[0]:
            # print("ko", lbd, x.min())
            return None

        # print("ok", lbd)
        outB = (c_b.T @ x).reshape(-1)
        outB[mask] = np.nan

        out = space.copy()
        out[:] = np.nan
        out[valid_limit] = outB

        # print(out)
        return out

    return evaluate