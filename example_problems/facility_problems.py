import os
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import json

from problems import Problem_sparse
from problems import solve
import pickle

all_problems = []
facility_folder = Path(__file__).parent / "facility"
for file in os.listdir(facility_folder):
    if os.path.isdir(facility_folder / file):
        all_problems.append(file)

def load_facility_problem(problem_name):
    folder = facility_folder / problem_name
    A_eq = sp.csr_array(sp.load_npz(folder / "A_eq.npz"))
    b_eq = np.load(folder / "b_eq.npy")

    a_1_ineq = sp.csr_array(sp.load_npz(folder / "A_ineq_1.npz"))
    b_1_ineq = np.load(folder / "b_ineq_1.npy")

    a_2_ineq = sp.csr_array(sp.load_npz(folder / "A_ineq_2.npz"))
    b_2_ineq = np.load(folder / "b_ineq_2.npy")

    c = np.load(folder / "c.npy")
    d_ineq = sp.csr_array(sp.load_npz(folder / "D_ineq.npz"))

    a_1_eq = sp.csr_matrix(A_eq)
    b_1_eq = b_eq.reshape((-1, 1))

    a_2_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, A_eq.shape[1]))
    b_2_eq = np.array([]).reshape((-1, 1))

    d_eq = sp.csr_matrix((np.array([]), (np.array([]), np.array([]))), shape=(0, A_eq.shape[1]))

    var_types = json.load(open(folder / "var_types.json"))

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
        c=c.reshape((-1, 1)),
        minimize=True,
        range=(0.7, 1),
        var_types=var_types)
    #problem.dual()

    return problem

if __name__ == '__main__':
    for p in sorted(all_problems):
        print(f"- facility_{p}")
        #solve(load_facility_problem(p), 1)