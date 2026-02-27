import os
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import json
import gurobipy

from problems import Problem_sparse, solve
from pulp import GUROBI

all_problems = []
miplib_folder = Path(__file__).parent / "uc"
for file in os.listdir(miplib_folder):
    if os.path.isdir(miplib_folder / file):
        all_problems.append(file)

def load_unit_problem(problem_name):
    folder = miplib_folder / problem_name

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
        range=(-1, 1),
        var_types=var_types)
    # problem.dual()
    #solve_gurobi(problem)
    return problem

def solve_gurobi(problem):
    m = gurobipy.Model()
    num_vars = problem.c.shape[0]
    vars = []
    for i in range(num_vars):
        if problem.var_types[i] == "C":
            v = m.addVar(vtype=gurobipy.GRB.CONTINUOUS, name=f"x_{i}")
        elif problem.var_types[i] == "B":
            v = m.addVar(vtype=gurobipy.GRB.BINARY, name=f"x_{i}")
        else:
            v = m.addVar(vtype=gurobipy.GRB.INTEGER, name=f"x_{i}")
        vars.append(v)

    m.update()
    # Objective
    obj = gurobipy.LinExpr()
    for i in range(num_vars):
        obj += problem.c[i, 0] * vars[i]

    m.setObjective(obj, gurobipy.GRB.MINIMIZE)

    # Constraints
    m.addMConstr(problem.a_1_eq, m.getVars(), gurobipy.GRB.EQUAL, problem.b_1_eq.flatten())
    m.addMConstr(problem.a_1_ineq, m.getVars(), gurobipy.GRB.LESS_EQUAL, problem.b_1_ineq.flatten())

    for i in range(problem.a_2_ineq.shape[0]):
        expr = gurobipy.LinExpr()
        for j in range(num_vars):
            coeff = problem.a_2_ineq[i, j]+problem.d_ineq[i, j]
            if coeff != 0:
                expr += coeff * vars[j]
        m.addConstr(expr <= problem.b_2_ineq[i, 0], name=f"ineq_{i}")

    m.optimize()


if __name__ == '__main__':
    for p in sorted(all_problems):
        problem = load_unit_problem(p)
        sol = solve(problem, -1)
        sol2 = solve(problem, 1)
        print(f"- unit_{p} {sol} {sol2}")