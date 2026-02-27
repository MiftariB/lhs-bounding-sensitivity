from __future__ import annotations

import os

import numpy as np
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import scipy.sparse as sp
import json
from os import listdir
from os.path import isfile, join

@dataclass
class CFLPInstance:
    m: int                           # number of facilities
    n: int                           # number of customers
    capacities: List[float]          # length m
    fixed_costs: List[float]         # length m
    demands: List[float]             # length n
    ship_costs: List[List[float]]    # m x n matrix, row-major per facility


def _tokenize(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for tok in f.read().split():
            yield tok


def parse_instance(path: str) -> CFLPInstance:
    toks = _tokenize(path)
    try:
        m = int(next(toks)); n = int(next(toks))
    except StopIteration:
        raise ValueError("File ended before reading m and n.")

    capacities, fixed_costs = [], []
    for i in range(m):
        try:
            cap = float(next(toks)); fix = float(next(toks))
        except StopIteration:
            raise ValueError(f"File ended while reading capacity/fixed cost for facility {i}.")
        capacities.append(cap); fixed_costs.append(fix)

    demands = []
    for j in range(n):
        try:
            demands.append(float(next(toks)))
        except StopIteration:
            raise ValueError(f"File ended while reading demand for customer {j}.")

    ship_costs: List[List[float]] = [[0.0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            try:
                ship_costs[i][j] = float(next(toks))
            except StopIteration:
                raise ValueError(f"File ended while reading shipping cost c[{i}][{j}].")

    # Optional sanity checks
    if any(d < 0 for d in demands):
        raise ValueError("Negative demand encountered.")
    if any(c < 0 for c in capacities):
        raise ValueError("Negative capacity encountered.")

    return CFLPInstance(m, n, capacities, fixed_costs, demands, ship_costs)


def build_cflp_matrices(n_facilities, n_customers, capacities, fixed_costs, demands, costs):
    # nb of facilities m
    #
    num_x = n_facilities * n_customers
    num_y = n_facilities
    num_vars = num_x + num_y

    get_x_idx = lambda i, j: i * n_customers + j
    get_y_idx = lambda i: num_x + i
    # c vector
    c = np.zeros(num_vars)
    for i in range(n_facilities):
        for j in range(n_customers):
            c[get_x_idx(i,j)] = costs[i][j]

    for i in range(n_facilities):
        c[get_y_idx(i)] = fixed_costs[i]

    # Equality constraints (demand satisfaction)
    A_eq = np.zeros((n_customers, num_vars))
    b_eq = np.ones((n_customers, 1))
    for j in range(n_customers):
        for i in range(n_facilities):
            A_eq[j, get_x_idx(i,j)] = 1.0  # coefficient for x_ij

    # Inequality constraints (capacity)
    A_ineq = np.zeros((n_facilities+num_vars, num_vars))
    D_ineq = np.zeros((n_facilities+num_vars, num_vars))
    b_ineq = np.zeros(n_facilities+num_vars)
    for i in range(n_facilities):
        for j in range(n_customers):
            A_ineq[i, get_x_idx(i,j)] = demands[j]  # x_ij
        if capacities[i] <= 100:
            A_ineq[i, get_y_idx(i)] = -capacities[i]
        else:
            D_ineq[i, get_y_idx(i)] = -capacities[i]

    for i in range(num_vars):
        A_ineq[n_facilities+i, i] = -1.0

    variables_out = ["C"] * num_x + ["B"] * num_y
    return sp.csr_matrix(A_eq), b_eq, sp.csr_matrix(A_ineq), b_ineq, sp.csr_matrix(D_ineq), c, variables_out

# Example export
mypath = "raw_files/"
print(os.getcwd())

output_path = "../facility/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    filename = file.split(".")[0]
    data = parse_instance(mypath + file)

    A_eq, b_eq, A_ineq, b_ineq, D_ineq, c, variables_out = build_cflp_matrices(data.m, data.n, data.capacities,
                                                            data.fixed_costs, data.demands, data.ship_costs)
    os.makedirs(output_path+filename)
    np.savez(output_path+filename+"/matrices.npz", A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, D_ineq=D_ineq, c=c, pickle=True)

    with open(output_path+filename+"/var_types.json", "w") as f:
        json.dump(variables_out, f)
