from pathlib import Path

import numpy as np
import os

from problems import Problem_sparse
import scipy.sparse as sp

""" Belgian 3 cluster model """

folder = Path(os.path.dirname(__file__)) / "data_3_cluster"

a_eq = sp.csr_array(sp.load_npz(folder / "a_eq.npz"))
b_eq = np.load(folder / "eq_b.npz")["arr_0"]

a_ineq = sp.csr_array(sp.load_npz(folder / "A_ineq_new.npz"))
b_ineq = np.load(folder / "b_ineq_new.npz")["arr_0"]

a2_ineq = sp.csr_array(sp.load_npz(folder / "A_2.npz"))
b2_ineq = np.load(folder / "b2.npz")["arr_0"]

c = np.load(folder / "objective.npz")["arr_0"].transpose()

a_1_ineq = a_ineq
a_1_eq = a_eq

b_1_eq = b_eq
b_1_ineq = b_ineq

a_2_ineq = a2_ineq
b_2_ineq = b2_ineq

nb_var = a_2_ineq.shape[1]

print(b_1_eq.shape)
print(c.shape)

# [685 497, 'e_consumed', 'continuous', 8760]
# [1 403 845, 'e_consumed', 'continuous', 8760]
# [1 903 183, 'e_consumed', 'continuous', 8760]

d_data = -np.ones(8760*3)
d_row = np.arange(0, 8760*3)
d_col = np.zeros(8760*3)
d_col[0:8760] = np.arange(685497, 685497+8760)
d_col[8760:2*8760] = np.arange(1403845, 1403845+8760)
d_col[2*8760:3*8760] = np.arange(1903183, 1903183+8760)

d_ineq = sp.csr_array(sp.coo_matrix((d_data, (d_row, d_col)), shape=a_2_ineq.shape))

d_eq = sp.csr_array(sp.coo_matrix(([], ([], [])), shape=(0, nb_var)))
a_2_eq = sp.csr_array(sp.coo_matrix(([], ([], [])), shape=(0, nb_var)))
b_2_eq = np.array([])

problem = Problem_sparse(
                a_1_eq=a_1_eq,
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
                minimize=True,
                range=(-0.85, 1-0.887))
problem.dual()