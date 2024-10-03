import functools
from dataclasses import dataclass

import scipy.sparse as sp
import numpy as np
from solveapi import solve_api

@dataclass
class Problem_sparse:
    """
    min/max c @ x
    a_1_eq @ x == b_1_eq
    a_1_ineq @ x <= b_1_ineq
    (a_2_eq + lbd * d_eq) @ x == b2_eq
    (a_2_ineq + lbd * d_ineq) @ x <= b2_ineq

    over given range.
    """

    # A_1 equalities
    a_1_eq: sp.csr_matrix
    b_1_eq: np.ndarray

    # A_1 inequalities
    a_1_ineq: sp.csr_matrix
    b_1_ineq: np.ndarray

    # A_2 equalities
    a_2_eq: sp.csr_matrix
    b_2_eq: np.ndarray
    d_eq: sp.csr_matrix

    # A_2 inequalities
    a_2_ineq: sp.csr_matrix
    b_2_ineq: np.ndarray
    d_ineq: sp.csr_matrix

    # Objective
    c: np.ndarray

    minimize: bool

    range: (float, float)

    _dual: "Problem_sparse" = None

    def __iter__(self):
        return iter(self.__dict__.values())

    def dual(self):
        if self._dual is None:

            dual_a_2_eq = sp.vstack([self.a_1_eq, self.a_1_ineq, self.a_2_eq, self.a_2_ineq], format="csr").transpose()
            dual_d_eq = sp.vstack([sp.csr_array(sp.coo_array(([], ([], [])), shape=(self.a_1_ineq.shape[0] +
                                                                                    self.a_1_eq.shape[0],
                                                                                    self.a_1_ineq.shape[1]))),
                                     self.d_eq, self.d_ineq], format="csr").transpose()
            dual_b_2_eq = self.c
            sign = 1 if self.minimize else -1

            first_part_nb_lines = self.a_1_ineq.shape[0]
            second_part_nb_lines = self.d_ineq.shape[0]

            first_part = sp.hstack([sp.csr_array(sp.coo_array(([], ([], [])), shape=(first_part_nb_lines,
                                                                                     self.a_1_eq.shape[0]))),
                                    sp.identity(self.a_1_ineq.shape[0]).multiply(sign),
                                    sp.csr_array(sp.coo_array(([], ([], [])), shape=(first_part_nb_lines,
                                                                                     self.d_eq.shape[0]))),
                                    sp.csr_array(sp.coo_array(([], ([], [])), shape=(first_part_nb_lines,
                                                                                     self.d_ineq.shape[0])))],
                                    format="csr")
            second_part = sp.hstack([sp.csr_array(sp.coo_array(([], ([], [])), shape=(second_part_nb_lines,
                                                                                      self.a_1_eq.shape[0]))),
                                     sp.csr_array(sp.coo_array(([], ([], [])), shape=(second_part_nb_lines,
                                                                                      self.a_1_ineq.shape[0]))),
                                     sp.csr_array(sp.coo_array(([], ([], [])), shape=(second_part_nb_lines,
                                                                                      self.d_eq.shape[0]))),
                                     sp.identity(self.d_ineq.shape[0]).multiply(sign)],
                                     format="csr")

            dual_a_1_ineq = sp.vstack([first_part, second_part], format="csr")
            dual_b_1_ineq = np.zeros(dual_a_1_ineq.shape[0])

            hot_2 = np.array((dual_d_eq != 0).sum(axis=1) != 0).reshape((-1,))
            nhot_2 = np.array((dual_d_eq != 0).sum(axis=1) == 0).reshape((-1,))
            dual_a_1_eq = dual_a_2_eq[nhot_2, :]
            dual_b_1_eq = dual_b_2_eq[nhot_2, :]

            dual_a_2_eq = dual_a_2_eq[hot_2, :]
            dual_d_eq = dual_d_eq[hot_2, :]
            dual_b_2_eq = dual_b_2_eq[hot_2, :]

            dual_a_2_ineq = sp.csr_array(sp.coo_array(([], ([], [])), shape=(0, dual_a_1_ineq.shape[1])))
            dual_d_ineq = sp.csr_array(sp.coo_array(([], ([], [])), shape=(0, dual_a_1_ineq.shape[1])))
            dual_b_2_ineq = np.array([])

            new_c = np.concatenate([self.b_1_eq, self.b_1_ineq, self.b_2_eq, self.b_2_ineq]).reshape(-1, 1)
            self._dual = Problem_sparse(
                a_1_eq=dual_a_1_eq,
                b_1_eq=dual_b_1_eq,
                a_1_ineq=dual_a_1_ineq,
                b_1_ineq=dual_b_1_ineq,
                a_2_eq=dual_a_2_eq,
                b_2_eq=dual_b_2_eq,
                d_eq=dual_d_eq,
                a_2_ineq=dual_a_2_ineq,
                b_2_ineq=dual_b_2_ineq,
                d_ineq=dual_d_ineq,
                c=new_c,
                minimize=not self.minimize,
                range=self.range,
                _dual=self
            )

        return self._dual

Problem = Problem_sparse

@functools.singledispatch
def solve(problem, lbd, c_2=None, debug=False, return_basis=False):
    raise Exception("not implemented")


@solve.register
def solve_sparse(problem: Problem, lbd, c_2=None, debug=False, return_basis=False):
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, _ = problem
    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(nb_var)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    api.add_constr(a_2_eq + d_eq * lbd, "==", b_2_eq.reshape(-1, ))
    api.add_constr(a_2_ineq + d_ineq * lbd, "<=", b_2_ineq.reshape(-1, ))

    api.set_obj(c.transpose(), "minimize" if mini else "maximize")
    api.optimize()
    status = api.get_status()
    if status != "unknown":
        return api.get_objective()
    else:
        return np.nan