import pickle

import gurobipy
from gurobipy import quicksum, GRB


def create_constraints(m, vars):

    # # g1
    non_none_vars = [
        m.addVar(vtype=GRB.BINARY, name=f"proxy_g1_{i}")
        for i, var in enumerate(vars)
        if i in section_names_idx
    ]
    for i, non_none_var in enumerate(non_none_vars):
        m.addConstr((non_none_var == 0) >> (vars[section_names_idx[i]] == 832))

    m.addConstr(vars[12893] == (quicksum(non_none_vars)))


    # g2
    m.addConstr(vars[13956] <= vars[10840], "g2")

    # g3, use temp var to force an integer value

    temp = m.addVar(vtype=GRB.INTEGER, name="temp1")
    # temp2 = m.addVar(vtype=GRB.INTEGER, name="temp2")
    # m.addConstr(temp >= 2)
    # m.addConstr(temp2 >= 2)
    m.addGenConstrLogA(vars[13956], temp, 2)


    # g4
    imports_vars = [var for i, var in enumerate(vars) if i in imports_idx]
    m.addConstr(quicksum(imports_vars) <= vars[271], "g4")

    # g5
    dll_imports_vars = [var for i, var in enumerate(vars) if i in dll_imports_idx]
    m.addConstr(quicksum(dll_imports_vars) <= vars[8607], "g5")

    # g6
    freq_idx_vars = [var for i, var in enumerate(vars) if i in freq_idx]
    m.addConstr(quicksum(freq_idx_vars) == 1, "g6")

    # g7
    freq_idx_vars = [var for i, var in enumerate(vars) if i in freq_idx]
    log_vars = [
        m.addVar(lb=-GRB.INFINITY, ub=0.0, vtype=GRB.CONTINUOUS, name=f"proxy_g7_{i}")
        for i, var in enumerate(vars)
        if i in freq_idx
    ]
    for i, log_var in enumerate(log_vars):
        m.addGenConstrLogA(vars[freq_idx[i]], log_var, 2)

    products = [freq_idx_vars[i] * log_vars[i] for i in range(len(freq_idx_vars))]
    m.addConstr(vars[23549] == -(quicksum(products)))
