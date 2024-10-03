import numpy as np
import scipy.sparse as sp

class NoSolution(Exception):
    pass

class GurobiModel:
    def __init__(self):
        from gurobipy import Model
        self.model = Model()
        self.var = None
        self.obj = []

    def add_var(self, nb_var):
        vars = self.model.addMVar(nb_var, lb=float('-inf'), ub=float('inf'))
        self.var = vars

    def add_constr(self, matrix, sense, rhs):
        from gurobipy import GRB
        gurobi_sense = GRB.EQUAL
        if sense == "<=":
            gurobi_sense = GRB.LESS_EQUAL
        elif sense == ">=":
            gurobi_sense = GRB.GREATER_EQUAL
        self.model.addMConstr(matrix, self.var, gurobi_sense, rhs)

    def set_obj(self, objective, sense):
        from gurobipy import GRB
        obj_sense = GRB.MINIMIZE
        if sense == "maximize":
            obj_sense = GRB.MAXIMIZE
        self.model.setObjective(objective @ self.var, obj_sense)
        self.obj = objective

    def optimize(self):
        self.model.optimize()

    def get_objective(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.model.getObjective()

    def get_dual(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.model.getAttr("Pi", self.model.getConstrs())

    def get_reduced_cost(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.model.getAttr("RC", self.model.getVars())

    def get_basis(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        basis_var = self.model.getAttr("VBasis", self.model.getVars())
        basis_var = [1 if (i == 0) else 0 for i in basis_var]
        basis_slack = self.model.getAttr("CBasis", self.model.getConstrs())
        basis_slack = [1 if (i == 0) else 0 for i in basis_slack]
        return np.array(basis_var), np.array(basis_slack)

    def get_solution(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.var.X

    def get_status(self):
        from gurobipy import GRB
        ret_status = self.model.status
        if ret_status == GRB.OPTIMAL:
            ret_status = "optimal"
        elif ret_status == GRB.SUBOPTIMAL:
            ret_status = "suboptimal"
        else:
            ret_status = "unknown"
        return ret_status


class CplexModel:
    def __init__(self):
        from cplex import Cplex
        self.model = Cplex()
        self.nb_var = 0
        self.nb_constr = 0
        self.disable_crossover = True

    def add_var(self, nb_var, lb=None, ub=None):
        import cplex
        self.nb_var = nb_var

        if lb is None:
            lb = [-cplex.infinity]*nb_var
        else:
            lb = [i if i is not None else -cplex.infinity for i in lb]

        if ub is None:
            ub = [cplex.infinity]*nb_var
        else:
            ub = [i if i is not None else cplex.infinity for i in lb]
        self.model.variables.add(lb=lb,
                                 ub=ub)

    def add_constr(self, matrix: sp.csr_array, sense: list, rhs: np.ndarray):
        nb_constr, _ = matrix.shape
        if nb_constr != 0:
            coo_mat = matrix.tocoo()
            coo_row, coo_col, coo_data = coo_mat.row, coo_mat.col, coo_mat.data
            coo_row += self.nb_constr
            if sense == "==":
                matrix_sense = "E"
            elif sense == "<=":
                matrix_sense = "L"
            else:
                matrix_sense = "G"

            self.model.linear_constraints.add(senses=[matrix_sense] * nb_constr, rhs=rhs.tolist()), max(coo_row)
            matrix_zipped = zip(coo_row.tolist(), coo_col.tolist(), coo_data.astype(float).tolist())

            self.model.linear_constraints.set_coefficients(matrix_zipped)
            self.nb_constr += nb_constr

    def set_obj(self, objective: np.ndarray, sense: str):
        pair_obj = [(i, val) for i, val in enumerate(objective.flatten())]
        self.model.objective.set_linear(pair_obj)
        if sense == "minimize":
            self.model.objective.set_sense(self.model.objective.sense.minimize)
        else:
            self.model.objective.set_sense(self.model.objective.sense.maximize)

    def optimize(self):
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.concurrent)
        if self.disable_crossover:
            self.model.parameters.solutiontype.set(2)
        else:
            self.model.parameters.solutiontype.set(0)
        #self.model.parameters.simplex.tolerances.optimality.set(1e-9)
        self.model.set_log_stream(None)
        self.model.set_results_stream(None)
        self.model.parameters.threads.set(5)
        self.model.parameters.timelimit.set(5400)
        #self.model.parameters.barrier.convergetol.set(1e-4)
        self.model.solve()

    def activate_crossover(self):
        self.disable_crossover = False

    def get_objective(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.model.solution.get_objective_value()

    def get_solution(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return np.array(self.model.solution.get_values())

    def get_dual(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return np.array(self.model.solution.get_dual_values())

    def get_reduced_cost(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return np.array(self.model.solution.get_reduced_costs())

    def get_basis(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        basis_var, basis_slack = self.model.solution.basis.get_basis()
        basis_var = [x == 1 for x in basis_var]
        basis_slack = [x == 1 for x in basis_slack]
        return np.array(basis_var, dtype=bool), np.array(basis_slack, dtype=bool)

    def get_free(self):
        basis_var, _ = self.model.solution.basis.get_basis()
        free_basis = [i == 3 for i in basis_var]
        return np.array(free_basis, dtype=bool)

    def get_status(self):
        status_code = self.model.solution.get_status()
        stopped = {6, 102, 10, 11, 12, 13, 107}
        if status_code == 1 or status_code == 101:
            ret_status = "optimal"
        elif status_code in stopped:
            ret_status = "suboptimal"
        else:
            ret_status = "unknown"
        return ret_status


class HiGHsModel:
    def __init__(self):
        from pyhighs import PyHighs
        self.pylib = PyHighs("/Users/bmiftari/Documents/inge21-22/HiGHS/HiGHS/build/lib/libhighs.dylib")
        self.model = self.pylib.Highs_create()
        self.nb_vars: int = 0
        self.matrix_eq: sp.csr_array | None = None
        self.matrix_ineq: sp.csr_array | None = None
        self.sense: list = []
        self.rhs_eq: np.ndarray = np.array([])
        self.rhs_ineq: np.ndarray = np.array([])
        self.obj: np.ndarray = np.array([])
        self.status: str = ""
        self.obj_sense: str = "minimize"
        self.solution: np.ndarray = np.array([])
        self.dual_values: np.ndarray = np.array([])
        self.obj_sol: int = 0

    def add_var(self, nb_var: int):
        self.nb_vars = nb_var

    def add_constr(self, matrix: sp.csr_array, sense: list, rhs: np.ndarray):
        if sense == "==":
            self._add_constr_eq(matrix, rhs)
        elif sense == "<=":
            self._add_constr_leq(matrix, rhs)
        elif sense == ">=":
            self._add_constr_beq(matrix, rhs)

    def _add_constr_eq(self, matrix: sp.csr_array, rhs: np.ndarray):
        if self.matrix_eq is None:
            self.matrix_eq = matrix
            self.rhs_eq = rhs
        else:
            self.matrix_eq = sp.vstack([self.matrix_eq, matrix], format="csr")
            self.rhs_eq = np.concatenate((self.rhs_eq, rhs))

    def _add_constr_leq(self, matrix: sp.csr_array, rhs: np.ndarray):
        if self.matrix_ineq is None:
            self.matrix_ineq = matrix
            self.rhs_ineq = rhs
        else:
            self.matrix_ineq = sp.vstack([self.matrix_ineq, matrix], format="csr")
            self.rhs_ineq = np.concatenate((self.rhs_ineq, rhs))

    def _add_constr_beq(self, matrix: sp.csr_array, rhs: np.ndarray):
        if self.matrix_ineq is None:
            self.matrix_ineq = -matrix
            self.rhs_ineq = -rhs
        else:
            self.matrix_ineq = sp.vstack([self.matrix_ineq, -matrix], format="csr")
            self.rhs_ineq = np.concatenate((self.rhs_ineq, -rhs))

    def set_obj(self, objective: np.ndarray, sense: str):
        self.obj_sense = sense
        self.obj = objective

    def optimize(self):
        self.pylib.Highs_setStringOptionValue(self.model, "solver", "ipm")
        self.pylib.Highs_setIntOptionValue(self.model, "threads", 10)
        self.pylib.Highs_setStringOptionValue(self.model, "parallel", "on")
        self.pylib.Highs_setBoolOptionValue(self.model, "run_crossover", False)
        self.pylib.Highs_setBoolOptionValue(self.model, "log_to_console", True)
        self.pylib.Highs_setIntOptionValue(self.model, "log_dev_level", 3)
        #self.pylib.Highs_setIntOptionValue(self.model, "ipm_iteration_limit", 10)

        nb_cols = self.nb_vars
        nb_constr_ineq = self.matrix_ineq.shape[0]
        nb_constr_eq = self.matrix_eq.shape[0]
        col_types = [0] * nb_cols
        col_lower = [-float('inf')] * nb_cols
        col_upper = [float('inf')] * nb_cols
        row_lower = [-float('inf')] * nb_constr_ineq + self.rhs_eq.tolist()
        row_upper = self.rhs_ineq.tolist() + self.rhs_eq.tolist()
        full_matrix = sp.vstack([self.matrix_ineq, self.matrix_eq])
        nb_rows = full_matrix.shape[0]
        mat_val, mat_row, mat_col = full_matrix.data, full_matrix.indptr, full_matrix.indices
        nb_values = len(mat_val)
        if self.obj_sense == "minimize":
            obj_sense = 1
        else:
            obj_sense = -1
        objective = self.obj.flatten()

        self.pylib.Highs_passLp(self.model, nb_cols,
                                nb_rows, nb_values, 2, obj_sense,
                                0, objective.tolist(), col_lower,
                                col_upper, row_lower, row_upper,
                                mat_row,
                                mat_col,
                                mat_val)
        self.pylib.Highs_run(self.model)
        status, self.solution,_, _, self.dual_values = self.pylib.Highs_getSolution(self.model)
        self.solution = np.array(self.solution)
        self.obj_sol = self.obj @ self.solution
        if status == 0:
            self.status = "optimal"
        else:
            self.status = "unknown"

    def get_objective(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return self.obj_sol

    def get_solution(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return np.array(self.solution)

    def get_dual(self):
        if self.get_status() not in ["optimal", "suboptimal"]:
            raise NoSolution()
        return np.array(self.dual_values)

    def get_status(self):
        return self.status


def solve_api(model_type):
    if model_type == "gurobi":
        return GurobiModel()
    elif model_type == "cplex":
        return CplexModel()
    elif model_type == "highs":
        return HiGHsModel()