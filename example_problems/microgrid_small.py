from pathlib import Path

import numpy as np
import scipy.sparse as sp
import os

from problems import Problem_sparse

""" Microgrid on a very limited number of time steps """

folder = Path(os.path.dirname(__file__)) / "data_microgrid_small"

a_1_eq = np.load(folder / "a_eq.npy")
a_1_eq = sp.csr_matrix(a_1_eq)
b_1_eq = np.load(folder / "eq_b.npy").reshape((-1, 1))

a_1_ineq = np.load(folder / "a_ineq.npy")
a_1_ineq = sp.csr_matrix(a_1_ineq)
b_1_ineq = np.load(folder / "ineq_b.npy").reshape((-1, 1))

a_2_ineq = np.load(folder / "a2_ineq.npy")
d = np.zeros(a_2_ineq.shape)
d[:, 146] = a_2_ineq[:, 146]
a_2_ineq[:, 146] = 0
a_2_ineq = sp.csr_matrix(a_2_ineq)
d_ineq = sp.csr_matrix(d)
b_2_ineq = np.load(folder / "ineq_b2.npy").reshape((-1, 1))

a_2_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, a_1_eq.shape[1]))
d_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, a_1_eq.shape[1]))
b_2_eq = np.array([]).reshape((-1, 1))

c = np.load(folder / "objective.npy").transpose()

minimize = True

problem = Problem_sparse(a_1_eq=a_1_eq,
                         b_1_eq=b_1_eq,
                         a_1_ineq=a_1_ineq,
                         b_1_ineq=b_1_ineq,
                         a_2_eq=a_2_eq,
                         b_2_eq=b_2_eq,
                         d_eq=d_eq,
                         a_2_ineq=a_2_ineq,
                         b_2_ineq=b_2_ineq,
                         d_ineq=d_ineq,
                         c=c,
                         minimize=minimize,
                         range=(0, 0.05))
problem.dual()