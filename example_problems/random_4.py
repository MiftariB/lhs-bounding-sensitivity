from problems import Problem_sparse
import numpy as np
import scipy.sparse as sp

""" Problem randomly generated to contain some spikes """

a_1_ineq = np.array([[3, 1],[0, -1]])

b_1_ineq = np.array([[3], [3]])

a_2_ineq = np.array([[-5, -2],
                     [ 1,  4]])

d_ineq = np.array([[-3, -2],
                   [-3,  0]])

b_2_ineq = np.array([[ 0], [-3]])

c = np.array([[-2], [-2]], dtype=float)

minimize = True

range = (-2, 2)

a_1_ineq = sp.csr_matrix(a_1_ineq)
a_2_ineq = sp.csr_matrix(a_2_ineq)
d_ineq = sp.csr_matrix(d_ineq)

a_1_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 2))
b_1_eq = np.array([]).reshape((-1, 1))

a_2_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, a_1_eq.shape[1]))
d_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, a_1_eq.shape[1]))
b_2_eq = np.array([]).reshape((-1, 1))

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
                         range=range)
problem.dual()
