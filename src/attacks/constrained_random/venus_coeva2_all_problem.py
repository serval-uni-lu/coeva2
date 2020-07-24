import copy

from pymoo.model.problem import Problem
import numpy as np

from .. import venus_constraints


class VenusProblem(Problem):
    def __init__(
        self, initial_state, weight, model, encoder, scaler, scale_objectives=True
    ):
        min_max = encoder.get_min_max_genetic()
        self.weight = weight
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.original = initial_state
        self.original_mm = self.scaler.transform([initial_state])[0]
        self.scale_objectives = scale_objectives
        super().__init__(n_var=15, n_obj=1, n_constr=10, xl=min_max[0], xu=min_max[1])

    def _evaluate(self, x, out, *args, **kwargs):

        x_ml = self.encoder.from_genetic_to_ml(self.original, x).astype("float64")

        f1 = 0

        constraints = venus_constraints.evaluate(x_ml)

        if self.scale_objectives:
            constraints = self.encoder.constraint_scaler.transform(constraints)

        out["F"] = np.column_stack([f1])
        out["G"] = constraints
