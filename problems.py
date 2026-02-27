import dataclasses
import functools
from dataclasses import dataclass
from typing import Tuple

import scipy.sparse as sp
import numpy as np

from solveapi import solve_api


def eliminate_variables_not_in_a(matrix_a: sp.csr_matrix, indexes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix_a = matrix_a.tocsc()  # Efficient column access
    valid_indexes = []
    invalid_indexes = []
    for idx in indexes:
        col = matrix_a[:, idx]
        if col.nnz > 0:  # nnz = number of non-zero elements
            valid_indexes.append(idx)
        else:
            invalid_indexes.append(idx)

    return np.array(valid_indexes), np.array(invalid_indexes)


def negate_columns(matrix_a: sp.csr_matrix, indexes: np.ndarray) -> sp.csr_matrix:
    A_csc = matrix_a.tocsc(copy=True)  # Copy to avoid modifying original

    for idx in indexes:
        start, end = A_csc.indptr[idx], A_csc.indptr[idx + 1]
        A_csc.data[start:end] *= -1  # Negate only the relevant values

    return A_csc.tocsr()

def extend_matrix_unconstraint_variables(matrix_a: sp.csr_matrix, indexes: np.ndarray) -> sp.csr_matrix:
    matrix_a = matrix_a.tocsc()  # Efficient column access
    new_cols = -matrix_a[:, indexes].tocsc()  # Negate the columns corresponding to indexes

    # Stack original and new columns horizontally
    extended = sp.hstack([matrix_a, new_cols], format='csr')
    return extended

def classify_variables(matrix_a: sp.csr_matrix, b: np.ndarray):
    """
    Classify variables in the problem defined by a @ x == b
    :param a: csr_matrix
    :param b: np.ndarray
    :return: (hot, nhot) where hot is the indices of the variables that are not zero in b and nhot are the others
    """
    positive_var_index = set()
    negative_var_index = set()
    unconstrainted_var_index = []
    indptr = matrix_a.indptr
    indices = matrix_a.indices
    data = matrix_a.data
    for i in range(matrix_a.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end - start != 1:
            continue  # Skip rows with more than one non-zero

        index = indices[start]
        val = data[start]

        if index in positive_var_index or index in negative_var_index:
            continue

        bi = b[i]
        if bi == 0:
            if val > 0:
                negative_var_index.add(index)
            elif val < 0:
                positive_var_index.add(index)
        elif bi < 0:
            if val < 0:
                positive_var_index.add(index)
            elif val > 0:
                negative_var_index.add(index)

    for i in range(matrix_a.shape[1]):
        if i not in positive_var_index and i not in negative_var_index:
            unconstrainted_var_index.append(i)

    positive_var_index = list(positive_var_index)
    negative_var_index = list(negative_var_index)

    positive_var_index.sort()
    negative_var_index.sort()

    unconstrainted_var_index.sort()
    return np.array(positive_var_index), np.array(negative_var_index), np.array(unconstrainted_var_index)





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

    var_types: list[str]|None = None

    _dual: "Problem_sparse" = None
    _positive: "Problem_sparse" = None

    def __iter__(self):
        return iter(self.__dict__.values())

    def linear(self):
        return dataclasses.replace(self, var_types=None, _positive=None, _dual=None)

    def positive(self):
        """return problem with only positive variables """
        if self._positive is not None:
            return self._positive
        
        if self.b_1_eq.ndim == 1:
            self.b_1_eq = self.b_1_eq.reshape((-1, 1))

        if self.b_1_ineq.ndim == 1:
            self.b_1_ineq = self.b_1_ineq.reshape((-1, 1))

        first_api = solve_api("cplex")
        first_api.add_var(self.a_1_eq.shape[1])
        first_api.add_constr(self.a_1_eq, "==", self.b_1_eq.reshape(-1, ))
        first_api.add_constr(self.a_1_ineq, "<=", self.b_1_ineq.reshape(-1, ))
        first_api.set_obj(self.c.transpose(), "minimize" if self.minimize else "maximize")
        lb_var, ub_var, _ = first_api.model.advanced.basic_presolve()

        neg_var = []
        positive_var = []
        unclass_var = []

        for i, (lb, ub) in enumerate(zip(lb_var, ub_var)):
            if ub <= 0 and lb < 0:
                neg_var.append(i)
            elif lb >= 0 and ub >= 0:
                positive_var.append(i)
            else:
                unclass_var.append(i)

        neg_var = np.array(neg_var, dtype=np.int_)
        positive_var = np.array(positive_var, dtype=np.int_)
        unclass_var = np.array(unclass_var, dtype=np.int_)
        nb_var = len(positive_var) + len(neg_var) + len(unclass_var)*2

        a_1_eq = negate_columns(self.a_1_eq, neg_var)
        a_1_eq = extend_matrix_unconstraint_variables(a_1_eq, unclass_var)

        a_1_ineq = negate_columns(self.a_1_ineq, neg_var)
        a_1_ineq = extend_matrix_unconstraint_variables(a_1_ineq, unclass_var)

        a_2_eq = negate_columns(self.a_2_eq, neg_var)
        a_2_eq = extend_matrix_unconstraint_variables(a_2_eq, unclass_var)

        d_eq = negate_columns(self.d_eq, neg_var)
        d_eq = extend_matrix_unconstraint_variables(d_eq, unclass_var)

        a_2_ineq = negate_columns(self.a_2_ineq, neg_var)
        a_2_ineq = extend_matrix_unconstraint_variables(a_2_ineq, unclass_var)

        d_ineq = negate_columns(self.d_ineq, neg_var)
        d_ineq = extend_matrix_unconstraint_variables(d_ineq, unclass_var)


        c = np.zeros((nb_var, 1))
        c[0:self.a_1_eq.shape[1]] = self.c
        c[neg_var] = -c[neg_var]
        if unclass_var.size > 0:
            c[self.a_1_eq.shape[1]:nb_var] = -self.c[unclass_var]

        var_types = self.var_types
        if var_types is not None:
            var_types = var_types + [var_types[i] for i in unclass_var]

        assert a_1_eq.shape[1] == nb_var
        assert a_1_ineq.shape[1] == nb_var
        assert a_2_eq.shape[1] == nb_var
        assert d_eq.shape[1] == nb_var
        assert a_2_ineq.shape[1] == nb_var
        assert d_ineq.shape[1] == nb_var
        assert c.shape[0] == nb_var
        assert len(var_types) == nb_var if var_types is not None else True
        
        out = Problem_sparse(
            a_1_eq=a_1_eq,
            b_1_eq=self.b_1_eq,
            a_1_ineq=sp.vstack([a_1_ineq, -sp.identity(nb_var)], format="csr"),
            b_1_ineq=np.concatenate([self.b_1_ineq,
                                     np.zeros((nb_var, 1))]),
            a_2_eq=a_2_eq,
            b_2_eq=self.b_2_eq,
            d_eq=d_eq,
            a_2_ineq=a_2_ineq,
            b_2_ineq=self.b_2_ineq,
            d_ineq=d_ineq,
            c=c,
            minimize=self.minimize,
            range=self.range,
            var_types=var_types
        )
        out._positive = out
        self._positive = out
        return out


    def dual(self):
        assert self.var_types is None or ("I" not in self.var_types and "B" not in self.var_types), "Dual not implemented for MIP"
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
            dual_a_1_eq = sp.csr_array(dual_a_2_eq[nhot_2, :])
            dual_b_1_eq = dual_b_2_eq[nhot_2, :].reshape(-1, 1)

            dual_a_2_eq = sp.csr_array(dual_a_2_eq[hot_2, :])
            dual_d_eq = sp.csr_array(dual_d_eq[hot_2, :])
            dual_b_2_eq = dual_b_2_eq[hot_2, :].reshape(-1, 1)

            dual_a_2_ineq = sp.csr_array(sp.coo_array(([], ([], [])), shape=(0, dual_a_1_ineq.shape[1])))
            dual_d_ineq = sp.csr_array(sp.coo_array(([], ([], [])), shape=(0, dual_a_1_ineq.shape[1])))
            dual_b_2_ineq = np.array([]).reshape(-1, 1)

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
def solve(problem, lbd, c_2=None, debug=False, return_basis=False, maxtime=None, fail_on_timeout=False):
    raise Exception("not implemented")


@solve.register
def solve_sparse(problem: Problem, lbd, c_2=None, debug=False, return_basis=False, maxtime=None, fail_on_timeout=False):
    a_1_eq, b_1_eq, a_1_ineq, b_1_ineq, a_2_eq, b_2_eq, d_eq, a_2_ineq, b_2_ineq, d_ineq, c, mini, _, var_types, _, _ = problem

    api = solve_api("cplex")
    nb_var = a_1_ineq.shape[1]
    api.add_var(nb_var, lb=[None]*nb_var, ub=([None]*nb_var), types=var_types)

    api.add_constr(a_1_eq, "==", b_1_eq.reshape(-1, ))
    api.add_constr(a_1_ineq, "<=", b_1_ineq.reshape(-1, ))

    api.add_constr(a_2_eq + d_eq * lbd, "==", b_2_eq.reshape(-1, ))
    api.add_constr(a_2_ineq + d_ineq * lbd, "<=", b_2_ineq.reshape(-1, ))

    api.set_obj(c.transpose(), "minimize" if mini else "maximize")
    api.optimize(maxtime=maxtime)
    status = api.get_status()
    if status != "unknown":
        if maxtime is not None and fail_on_timeout and status != "optimal":
            return np.nan
        return api.get_objective()
    else:
        return np.nan