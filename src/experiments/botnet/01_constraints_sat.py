#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1py
#        x, y, z binary
from pathlib import Path

import gurobipy as gp
import joblib
import numpy as np
from gurobipy import GRB, QuadExpr, LinExpr
from tqdm import tqdm

from src.examples.malware.malware_constraints import MalwareConstraints
from src.utils import in_out, filter_initial_states
from src.examples.malware.malware_constraints_sat import create_constraints

type_mask_transform = {"real": GRB.CONTINUOUS, "int": GRB.INTEGER}
config = in_out.get_parameters()
scaler = joblib.load(config["paths"]["min_max_scaler"])


def create_l2_constraints(m, vars, x_init, lb, ub, l2):

    x_init_scaled = scaler.transform(x_init.reshape(1, -1))[0]

    # scaled_var = [
    #     m.addVar(
    #         lb=-2,
    #         ub=2,
    #         vtype=GRB.CONTINUOUS,
    #         name=f"scaled_f_{i}",
    #     )
    #     for i in range(len(vars))
    # ]

    expr = LinExpr()
    for i, e in enumerate(vars):
        m.addConstr(
            ((vars[i] - lb[i]) / (ub[i] - lb[i])) <= x_init_scaled[i] + 0.1,
            f"scaled_{i}",
        )
        m.addConstr(
            ((vars[i] - lb[i]) / (ub[i] - lb[i])) >= x_init_scaled[i] - 0.1,
            f"scaled_{i}",
        )
        # expr.add(scaled_var[i])

    # print(expr)
    # m.addConstr(expr <= 0.1)
    # m.setObjective(expr, sense=GRB.MINIMIZE)


def create_variable(m, x_init, type_mask, lb, ub):

    return [
        m.addVar(lb=lb[i], ub=ub[i], vtype=type_mask_transform[type_mask[i]], name=f"f{i}")
        for i, feature in enumerate(x_init)
    ]


def create_mutable_constraints(m, x_init, vars, mutable_mask):
    indexes = np.argwhere(~mutable_mask).reshape(-1)

    for i in indexes:
        m.addConstr(vars[i] == x_init[i], f"mut{i}")


def create_model(x_init, mutable_mask, type_mask, lb, ub, l2):
    m = gp.Model("mip1")
    vars = create_variable(m, x_init, type_mask, lb, ub)

    create_mutable_constraints(m, x_init, vars, mutable_mask)
    create_constraints(m, vars)
    create_l2_constraints(m, vars, x_init, lb, ub, l2)

    return m


def apply_on_single(x_init, mutable_mask, type_mask, lb, ub, l2):
    try:
        m = create_model(x_init, mutable_mask, type_mask, lb, ub, l2)
        m.setParam(GRB.Param.PoolSolutions, config["n_repetition"])
        m.setParam(GRB.Param.PoolSearchMode, 2)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.NumericFocus, 3)
        m.optimize()
        nSolutions = m.SolCount

        def get_vars(e):
            m.setParam(GRB.Param.SolutionNumber, e)
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


def apply_model(x_initial, constraints, lb, up, l2):
    mutable_mask = constraints.get_mutable_mask()
    type_mask = constraints.get_feature_type()

    iterable = x_initial
    if config["verbose"] > 0:
        iterable = tqdm(iterable, total=len(x_initial))

    return np.array(
        [
            apply_on_single(x_init, mutable_mask, type_mask, lb, up, l2)
            for x_init in iterable
        ]
    )


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    lb, ub = scaler.data_min_, scaler.data_max_
    ub[(ub - lb) == 0] = ub[(ub - lb) == 0] + 0.0000001

    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )

    l2_max = config["thresholds"]["f2"]

    constraints = MalwareConstraints(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    x_attacks = apply_model(x_initial, constraints, lb, ub, config["thresholds"]["f2"])
    # print(x_attacks[0][0][1017])
    print(x_attacks.shape)
    np.save(config["paths"]["attack_results"], x_attacks)


if __name__ == "__main__":
    run()
