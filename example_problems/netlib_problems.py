import os
from pathlib import Path
import scipy.sparse as sp
import numpy as np

from problems import Problem_sparse

all_problems = []
netlib_folder = Path(__file__).parent / "netlib"
for file in os.listdir(netlib_folder):
    if os.path.isdir(netlib_folder / file):
        all_problems.append(file)

def load_netlib_problem(problem_name):
    folder = netlib_folder / problem_name
    a_1_eq = sp.csr_array(sp.load_npz(folder / "a_1_eq.npz"))
    b_1_eq = np.load(folder / "b_1_eq.npz")["arr_0"]
    a_2_eq = sp.csr_array(sp.load_npz(folder / "a_2_eq.npz"))
    b_2_eq = np.load(folder / "b_2_eq.npz")["arr_0"]
    a_1_ineq = sp.csr_array(sp.load_npz(folder / "a_1_ineq.npz"))
    b_1_ineq = np.load(folder / "b_1_ineq.npz")["arr_0"]
    a_2_ineq = sp.csr_array(sp.load_npz(folder / "a_2_ineq.npz"))
    b_2_ineq = np.load(folder / "b_2_ineq.npz")["arr_0"]
    d_eq = sp.csr_array(sp.load_npz(folder / "d_eq.npz"))
    d_ineq = sp.csr_array(sp.load_npz(folder / "d_ineq.npz"))
    c = np.load(folder / "c.npz")["arr_0"]

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
        c=c.transpose(),
        minimize=True,
        range=(-1, 1))
    problem.dual()

    return problem

if __name__ == '__main__':
    for p in sorted(all_problems):
        print(f"- {p}")