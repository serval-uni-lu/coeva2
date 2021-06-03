import numpy as np
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB, QuadExpr, LinExpr
from src.attacks.moeva2.constraints import Constraints
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints

type_mask_transform = {
    "real": GRB.CONTINUOUS,
    "int": GRB.INTEGER,
    "ohe0": GRB.INTEGER,
    "ohe1": GRB.INTEGER,
    "ohe2": GRB.INTEGER,
}

SAFETY_DELTA = 0.0000001


class SatAttack:
    def __init__(
        self,
        constraints: Constraints,
        sat_constraints,
        min_max_scaler,
        eps,
        norm,
        n_sample=1,
        verbose=1,
    ):
        self.constraints = constraints
        self.sat_constraints = sat_constraints
        self.min_max_scaler = min_max_scaler
        self.eps = eps
        self.norm = norm
        self.n_sample = n_sample
        self.verbose = verbose
        self.encoder = get_encoder_from_constraints(self.constraints)

    def create_variable(self, m, x_init, type_mask, lb, ub):

        return [
            m.addVar(
                lb=lb[i],
                ub=ub[i],
                vtype=type_mask_transform[type_mask[i]],
                name=f"f{i}",
            )
            for i, feature in enumerate(x_init)
        ]

    def create_mutable_constraints(self, m, x_init, vars, mutable_mask):
        indexes = np.argwhere(~mutable_mask).reshape(-1)

        for i in indexes:
            m.addConstr(vars[i] == x_init[i], f"mut{i}")

    def create_l_constraints(self, m, vars, x_init, lb, ub):

        x_init_scaled = self.min_max_scaler.transform(x_init.reshape(1, -1))[0]

        expr = LinExpr()
        for i, e in enumerate(vars):
            if lb[i] != ub[i]:
                scaled = (vars[i] - lb[i]) / (ub[i] - lb[i])
                m.addConstr(
                    scaled <= x_init_scaled[i] + self.eps - SAFETY_DELTA,
                    f"scaled_{i}",
                )
                m.addConstr(
                    scaled >= x_init_scaled[i] - self.eps + SAFETY_DELTA,
                    f"scaled_{i}",
                )
            else:
                scaled = 0

    def apply_hot_start(self, vars, x_hot_start=None):
        if x_hot_start is not None:
            for i in range(len(vars)):
                vars[i].start = x_hot_start[i]

    def create_model(self, x_init, x_hot_start=None):
        # Pre fetch
        mutable_mask = self.constraints.get_mutable_mask()
        type_mask = self.constraints.get_feature_type()
        lb, ub = self.constraints.get_feature_min_max(dynamic_input=x_init)

        m = gp.Model("mip1")

        # Create variables
        vars = self.create_variable(m, x_init, type_mask, lb, ub)

        # Mutable constraints
        self.create_mutable_constraints(m, x_init, vars, mutable_mask)

        # Constraints
        self.sat_constraints(m, vars)

        # Distance constraints
        self.create_l_constraints(
            m,
            vars,
            x_init,
            self.min_max_scaler.data_min_,
            self.min_max_scaler.data_max_,
        )

        # Hot start
        self.apply_hot_start(vars, x_hot_start)

        return m

    def _one_generate(self, x_init, x_hot_start=None):
        try:

            m = self.create_model(x_init, x_hot_start)
            m.setParam(GRB.Param.PoolSolutions, self.n_sample)
            m.setParam(GRB.Param.PoolSearchMode, 2)
            m.setParam(GRB.Param.NonConvex, 2)
            m.setParam(GRB.Param.NumericFocus, 3)
            m.setParam(GRB.Param.StartNodeLimit, 1000)
            m.optimize()
            nSolutions = m.SolCount

            def get_vars(e):
                m.setParam(GRB.Param.SolutionNumber, e)
                # for v in m.getVars():
                #     print(v)
                return [v.X for v in m.getVars()]

            solutions = np.array([get_vars(e) for e in range(nSolutions)])[
                :, : x_init.shape[0]
            ]

            # print(solutions.shape)

        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))

        except AttributeError:
            print("Encountered an attribute error")

        return solutions

    def generate(self, x_initial, x_hot_start=None):

        x_hot_start_local = [None for i in range(x_initial.shape[0])]
        if x_hot_start is not None:
            if x_initial.shape != x_hot_start.shape:
                raise ValueError

            x_hot_start_local = x_hot_start

        iterable = enumerate(x_initial)
        if self.verbose > 0:
            iterable = tqdm(iterable, total=len(x_initial))

        return np.array(
            [self._one_generate(x_init, x_hot_start_local[i]) for i, x_init in iterable]
        )
