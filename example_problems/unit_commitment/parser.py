# parse_uc.py
import re
import numpy as np
from pathlib import Path


from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
import json
import os
from os import listdir
from os.path import isfile, join

# -----------------------------
# Thermal units
# -----------------------------
@dataclass
class ThermalUnit:
    id: int
    qcost: float        # quadratic cost coeff
    lcost: float        # linear cost coeff
    ccost: float        # constant cost coeff
    pmin: float
    pmax: float
    init_status: int    # initial on/off duration
    min_up: int
    min_down: int
    coolAndFuelCost: float
    hotAndFuelCost: float
    tau: float
    tauMax: float
    fixedCost: float
    p0: Optional[float] = None
    ramp_up: Optional[float] = None
    ramp_down: Optional[float] = None


# -----------------------------
# Hydro units
# -----------------------------
@dataclass
class HydroUnit:
    id: int
    volumeToPower: float  # conversion factor
    b_h: float          # unused
    maxUsage: float      # max water usage
    maxSillage: float    # max water spillage
    initialFlood: float  # initial flood volume
    minFlood: float      # min flood volume
    maxFlood: float      # max flood volume
    breakFlow: list[float] = field(default_factory=list)  # flow breakpoints


# -----------------------------
# Cascade links
# -----------------------------
@dataclass
class CascadeLink:
    id: float
    nb_hydro_unit: int
    hydro_units: List[HydroUnit] = field(default_factory=list)


# -----------------------------
# Complete UC problem
# -----------------------------
@dataclass
class UCProblem:
    horizon: int
    num_thermal: int
    num_hydro: int = 0
    num_cascades: int = 0

    loads: np.ndarray = field(default_factory=lambda: np.array([]))
    reserve: Optional[np.ndarray] = None

    thermal_units: List[ThermalUnit] = field(default_factory=list)
    hydro_units: List[HydroUnit] = field(default_factory=list)
    cascades: List[CascadeLink] = field(default_factory=list)

    # Optional: keep raw sections if needed
    raw_sections: dict = field(default_factory=dict)


def tokenize_file(path):
    with open(path, 'r') as f:
        lines = [ln.rstrip() for ln in f]
    # normalize whitespace, keep blank lines
    return lines

def floatable(s):
    try:
        float(s)
        return True
    except:
        return False

def parse_txt_tokens(filename: str):
    tokens = []
    with open(filename, "r") as f:
        for line in f:
            # strip trailing newline
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # split on whitespace (handles spaces & tabs)
            parts = line.split()
            tokens.append(parts)
    return tokens

def parse_format_file(path):
    tokens = parse_txt_tokens(path)
    problem_num = int(tokens[0][1])
    horizon = int(tokens[1][1])
    num_thermal = int(tokens[2][1])
    num_hydro = int(tokens[3][1])
    num_cascades = int(tokens[4][1])
    min_sys_capacity = float(tokens[6][1])
    max_sys_capacity = float(tokens[7][1])
    max_thermal_capacity = float(tokens[8][1])

    #removing headers
    tokens = tokens[9:]
    #print(tokens)
    _, nb_days_loads, nb_step_day = tokens.pop(0)
    nb_days_loads = int(nb_days_loads)
    nb_step_day = int(nb_step_day)
    loads = []
    for i in range(nb_days_loads):
        load_day = tokens.pop(0)
        loads.append([float(x) for x in load_day])

    nb_spinning_reserve = int(tokens.pop(0)[1])
    reserve = [float(i) for i in tokens.pop(0)]
    #remove header
    tokens.pop(0)
    current_termal_data = tokens.pop(0)
    termal_units = []
    for i in range(num_thermal):
        thermal_data = current_termal_data
        if len(thermal_data) < 15:
            raise ValueError(f"Thermal unit data line has insufficient fields: {thermal_data}")

        if tokens[0][0] == "RampConstraints":
            ramping_data = tokens.pop(0)
        else:
            ramping_data = [None, None, None]

        tu = ThermalUnit(
            id=int(thermal_data[0]),
            qcost=float(thermal_data[1]),
            lcost=float(thermal_data[2]),
            ccost=float(thermal_data[3]),
            pmin=float(thermal_data[4]),
            pmax=float(thermal_data[5]),
            init_status=int(thermal_data[6]),
            min_up=int(thermal_data[7]),
            min_down=int(thermal_data[8]),
            coolAndFuelCost=float(thermal_data[9]),
            hotAndFuelCost=float(thermal_data[10]),
            tau=float(thermal_data[11]),
            tauMax=float(thermal_data[12]),
            fixedCost=float(thermal_data[13]),
            p0=float(thermal_data[15]),
            ramp_up= float(ramping_data[1]) if ramping_data[1] is not None else None,
            ramp_down= float(ramping_data[2]) if ramping_data[2] is not None else None,
        )
        current_termal_data=tokens.pop(0)
        termal_units.append(tu)

    # hydrosection has been removed in current_termal_data
    hydro_units = []
    for i in range(num_hydro):
        hydro_data = tokens.pop(0)
        flows_data = [float(x) for x in tokens.pop(0)]
        hu = HydroUnit(
            id=int(hydro_data[0]),
            volumeToPower=float(hydro_data[1]),
            b_h=float(hydro_data[2]),
            maxUsage=float(hydro_data[3]),
            maxSillage=float(hydro_data[4]),
            initialFlood=float(hydro_data[5]),
            minFlood=float(hydro_data[6]),
            maxFlood=float(hydro_data[7]),
            breakFlow=flows_data
        )
        hydro_units.append(hu)

    return UCProblem(
        horizon=horizon,
        num_thermal=num_thermal,
        num_hydro=num_hydro,
        num_cascades=num_cascades,
        loads=np.array(loads).flatten(),
        reserve=np.array(reserve) if reserve else None,
        thermal_units=termal_units,
        hydro_units=hydro_units,
        cascades=[],
        raw_sections={}
    )


def build_milp_matrices(problem: UCProblem) \
        -> Tuple[Tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray, np.ndarray, Dict], List[str]]:
    m = gp.Model()
    T = problem.horizon
    U = problem.num_thermal
    units = problem.thermal_units
    H = problem.num_hydro
    hydros = problem.hydro_units

    # Decision variables
    # status of termal units
    start_i_t = m.addVars(U, T+1, vtype=gp.GRB.INTEGER, name="start")
    u_i_t = m.addVars(U, T+1, vtype=gp.GRB.INTEGER, name="u")
    # power of termal units
    p_i_t = m.addVars(U, T+1, vtype=gp.GRB.CONTINUOUS, name="p")

    # discharge water
    q_h_t = m.addVars(H, T+1, vtype=gp.GRB.CONTINUOUS, name="q")
    # reservoir volume
    v_h_t = m.addVars(H, T+1, vtype=gp.GRB.CONTINUOUS, name="v")
    # spilled water
    w_h_t = m.addVars(H, T+1, vtype=gp.GRB.CONTINUOUS, name="w")

    for i, u in enumerate(units):
        for t in range(T+1):
                m.addConstr(u_i_t[i, t] <= 1, name=f"u1_{i}_{t}")
                m.addConstr(u_i_t[i, t] >= 0, name=f"u0_{i}_{t}")

                m.addConstr(start_i_t[i, t] <= 1, name=f"start1_{i}_{t}")
                m.addConstr(start_i_t[i, t] >= 0, name=f"start0_{i}_{t}")

                m.addConstr(p_i_t[i, t] <= u.pmax * u_i_t[i, t], name=f"pmax_{i}_{t}")
                m.addConstr(p_i_t[i, t] >= u.pmin * u_i_t[i, t], name=f"pmin_{i}_{t}")

    for i, u in enumerate(units):
        for t in range(u.init_status):
            if u.init_status > 0:
                m.addConstr(u_i_t[i, t] == 1, name=f"init_up_{i}_{t}")
            else:
                m.addConstr(u_i_t[i, t] == 0, name=f"init_down_{i}_{t}")

    for i, u in enumerate(units):
        for t in range(1, T+1):
            m.addConstr(p_i_t[i, t] <= p_i_t[i, t-1] + u.pmax, name=f"pmax_{i}_{t}")
            m.addConstr(p_i_t[i, t-1] <= p_i_t[i, t] + u.pmin, name=f"pmin_{i}_{t}")

    for i, u in enumerate(units):
        for t in range(T+1):
            for r in range(max(1, t-u.min_up), max(1, t)):
                m.addConstr(u_i_t[i, t] >= u_i_t[i, r] - u_i_t[i, r-1], name=f"minup_{i}_{t}_{r}")

            for r in range(max(1, t - u.min_down), max(1, t)):
                m.addConstr(u_i_t[i, r] >= 1- u_i_t[i, r-1] - u_i_t[i, r], name=f"minup2_{i}_{t}_{r}")

    for h, hu in enumerate(hydros):
        for t in range(T+1):
            m.addConstr(v_h_t[h, t] <= hu.maxFlood, name=f"vmax_{h}_{t}")
            m.addConstr(v_h_t[h, t] >= hu.minFlood, name=f"vmin_{h}_{t}")
            m.addConstr(q_h_t[h, t] <= hu.maxUsage, name=f"qmax_{h}_{t}")
            m.addConstr(q_h_t[h, t] >= 0, name=f"qmin_{h}_{t}")
            m.addConstr(w_h_t[h, t] <= hu.maxSillage, name=f"wmax_{h}_{t}")
            m.addConstr(w_h_t[h, t] >= 0, name=f"wmin_{h}_{t}")

        m.addConstr(v_h_t[h, 0] == hu.initialFlood, name=f"v_init_{h}")

        for t in range(1, T+1):
            m.addConstr(v_h_t[h, t] - v_h_t[h, t-1] == - q_h_t[h, t] - w_h_t[h, t], name=f"v_balance_{h}_{t}")

    for t in range(1, T+1):
        m.addConstr(gp.quicksum(p_i_t[i, t] for i in range(U))
                    + gp.quicksum(hu.volumeToPower * q_h_t[h, t] for h, hu in enumerate(hydros)) >= problem.loads[t-1],
                    name=f"load_balance_{t}")

    for i, u in enumerate(units):
        for t in range(1, T + 1):
            m.addConstr(start_i_t[i, t] >= u_i_t[i, t] - u_i_t[i, t - 1] if t > 0 else u_i_t[i, t],
                        name=f"start_def_{i}_{t}")

    # Objective: minimize total cost
    obj = (gp.quicksum((u.lcost+u.qcost) * p_i_t[i, t] + u.ccost*u_i_t[i, t] for i, u in enumerate(units) for t in range(T+1))
           + gp.quicksum(u.fixedCost * start_i_t[i, t] for i, u in enumerate(units) for t in range(T+1)))

    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.update()
    m.optimize()

    # Extract matrices
    A_eq = []
    b_eq = []
    A_ineq_1 = []
    b_ineq_1 = []
    A_ineq_2 = []
    b_ineq_2 = []
    D_ineq = []

    var_list = list(m.getVars())
    var_index = {var.varName: idx for idx, var in enumerate(var_list)}
    num_vars = len(var_list)

    for constr in m.getConstrs():
        row = np.zeros(num_vars)
        chg_row = np.zeros(num_vars)

        is_lb = "load_balance" in constr.ConstrName
        expr = m.getRow(constr)  # LinExpr

        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)

            idx = var_index[var.VarName]

            if is_lb and "p" in var.VarName:
                chg_row[idx] = coeff*np.random.uniform(-0.3, 0.3)

            row[idx] = coeff

        if constr.Sense == gp.GRB.EQUAL:
            A_eq.append(row)
            b_eq.append(constr.RHS)
        elif constr.Sense == gp.GRB.LESS_EQUAL:
            if is_lb:
                D_ineq.append(chg_row)
                A_ineq_2.append(row)
                b_ineq_2.append(constr.RHS)
            else:
                A_ineq_1.append(row)
                b_ineq_1.append(constr.RHS)
        elif constr.Sense == gp.GRB.GREATER_EQUAL:
            if is_lb:
                D_ineq.append(-chg_row)
                A_ineq_2.append(-row)
                b_ineq_2.append(-constr.RHS)
            else:
                A_ineq_1.append(-row)
                b_ineq_1.append(-constr.RHS)

    A_eq = sp.csr_matrix(np.array(A_eq)) if A_eq else sp.csr_matrix((0, num_vars))
    b_eq = np.array(b_eq).reshape((-1, 1)) if b_eq else np.array([]).reshape((-1, 1))
    A_ineq_1 = sp.csr_matrix(np.array(A_ineq_1)) if A_ineq_1 else sp.csr_matrix((0, num_vars))
    neg_I = -sp.identity(num_vars, format="csr")
    A_ineq_1 = sp.vstack([A_ineq_1, neg_I], format="csr")

    b_ineq_1 = np.array(b_ineq_1).reshape((-1, 1)) if b_ineq_1 else np.array([]).reshape((-1, 1))
    zeros_to_add = np.zeros((num_vars, 1))
    b_ineq_1 = np.vstack([b_ineq_1, zeros_to_add])

    A_ineq_2 = sp.csr_matrix(np.array(A_ineq_2)) if A_ineq_2 else sp.csr_matrix((0, num_vars))
    b_ineq_2 = np.array(b_ineq_2).reshape((-1, 1)) if b_ineq_2 else np.array([]).reshape((-1, 1))

    D_ineq = sp.csr_matrix(np.array(D_ineq)) if D_ineq else sp.csr_matrix((0, num_vars))
    c = np.array([var.Obj for var in var_list]).reshape((-1, 1))

    chg_var = lambda x: 'B' if x == gp.GRB.BINARY else ('I' if x == gp.GRB.INTEGER else 'C')

    var_type = [chg_var(var.VType) for var in m.getVars()]

    matrices = {
        'A_Eq': A_eq,
        'b_eq': b_eq,
        'A_ineq_1': A_ineq_1,
        'b_ineq_1': b_ineq_1,
        'A_ineq_2': A_ineq_2,
        'b_ineq_2': b_ineq_2,
        'D_ineq': D_ineq,
        'c': c
    }

    return matrices, var_type


def save_matrices(mats, out_prefix):
    sp.save_npz(out_prefix + 'A_eq', mats['A_Eq'])
    np.save(out_prefix + 'b_eq', mats['b_eq'])

    sp.save_npz(out_prefix + 'A_ineq_1', mats['A_ineq_1'])
    np.save(out_prefix + 'b_ineq_1', mats['b_ineq_1'])

    sp.save_npz(out_prefix + 'A_ineq_2', mats['A_ineq_2'])
    np.save(out_prefix + 'b_ineq_2', mats['b_ineq_2'])
    sp.save_npz(out_prefix + 'D_ineq', mats['D_ineq'])

    np.save(out_prefix + 'c', mats['c'])



def main(filename, out_prefix):
    p = Path(filename)
    if not p.exists():
        print(f"File {filename} not found.")
        return
    parsed = parse_format_file(filename)
    print("Parsed summary:")
    print(f" Horizon: {parsed.horizon}, NumThermal: {parsed.num_thermal}, Thermal units parsed: {len(parsed.thermal_units)}")
    mats, variables_out = build_milp_matrices(parsed)
    #print("Matrix shapes:")
    #print(" A_Eq:", mats['A_Eq'].shape, " b_eq:", mats['b_eq'].shape)
    #print(" A_ineq:", mats['A_ineq'].shape, " b_ineq:", mats['b_ineq'].shape)
    #print(" c:", mats['c'].shape)
    print(variables_out)
    save_matrices(mats, out_prefix)
    with open(out_prefix + "var_types.json", "w") as f:
        json.dump(variables_out, f)
    #print(f"Matrices saved to {out_prefix}.npz and individual .npy files")

if __name__ == '__main__':
    # Example export
    mypath = "raw_files/"
    print(os.getcwd())

    output_path = "../uc/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        filename = file.split(".")[0]
        os.makedirs(output_path + filename)
        main(filename=mypath + file, out_prefix=output_path + filename + "/")

