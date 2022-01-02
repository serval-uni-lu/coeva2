from gurobipy import LinExpr


def sum_linear(feats, index_a):
    expr1 = LinExpr()
    for i in index_a:
        expr1.add(feats[i])
    return expr1


def create_constraints(m, feats):
    def apply_if_a_supp_zero_than_b_supp_zero(a, b):
        m.addConstr(feats[a] >= 1 >> feats[b] >= 1)

    # g1
    m.addConstr(feats[1] <= feats[0])

    # g2
    m.addConstr(sum_linear(feats, range(3, 18)) + 3 * feats[19] <= feats[0])

    # g3
    apply_if_a_supp_zero_than_b_supp_zero(21, 3)

    # g4
    apply_if_a_supp_zero_than_b_supp_zero(23, 13)

    # g5
    m.addConstr(
        3 * feats[20] + 4 * feats[21] + 4 * feats[22] + 2 * feats[23] <= feats[0]
    )

    # g6
    apply_if_a_supp_zero_than_b_supp_zero(19, 25)

    # g7 is not considered

    # g8
    apply_if_a_supp_zero_than_b_supp_zero(2, 25)

    # g9 is not considered

    # g10
    apply_if_a_supp_zero_than_b_supp_zero(28, 25)

    # g11
    apply_if_a_supp_zero_than_b_supp_zero(31, 26)

    # g12
    m.addConstr(feats[38] <= feats[37])

    # g13
    m.addConstr(3 * feats[20] <= feats[0] + 1)

    # g14
    m.addConstr(4 * feats[21] <= feats[0] + 1)

    # g15
    m.addConstr(4 * feats[22] <= feats[0] + 1)

    # g16
    m.addConstr(2 * feats[23] <= feats[0] + 1)
