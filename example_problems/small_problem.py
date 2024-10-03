"""
The problem is written as :
min 2x+2y
s.t -2x+2y <= 4
    -x <= 1
    2x + y + lambda (-x -4y) <= 4
    -2x -3y + lambda(4y) <= 2
    2x+2y+lambda(-4x-3y) <= 0
    -x-4y+lambda (2x+4y) <= 2
"""
import numpy as np
import scipy.sparse as sp
from problems import Problem_sparse


minimize = True
c = np.array([[2], [-2]])
a_1_ineq = sp.csr_matrix(np.array([[-2, 2], [-1, 0]]))
b_1_ineq = np.array([4, 1])

a_1_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 2))
b_1_eq = np.array([])

a_2_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 2))
d_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, 2))
b_2_eq = np.array([])

a_2_ineq = sp.csr_matrix(np.array([[2,  1], [-2, -3], [2,  2], [-1, -4]]))
d_ineq = sp.csr_matrix(np.array([[-1, -4], [0,  4], [-4, -3], [2,  4]]))
b_2_ineq = np.array([4, 2, 0, 2])

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
                         range=(-10, 9))
problem.dual()

