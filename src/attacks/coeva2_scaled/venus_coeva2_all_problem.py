import copy

from pymoo.model.problem import Problem
import numpy as np

from .. import venus_constraints


class VenusProblem(Problem):
    def __init__(
        self, initial_state, weight, model, encoder, scaler, scale_objectives=True, save_history=False
    ):
        min_max = encoder.get_min_max_genetic()
        self.weight = weight
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.original = initial_state
        self.original_mm = self.scaler.transform([initial_state])[0]
        self.scale_objectives = scale_objectives
        self.save_history = save_history
        self.history = {
            "f1": [],
            "f2": [],
            "f3": [],
            "g1": [],
        }
        super().__init__(n_var=15, n_obj=1, n_constr=10, xl=min_max[0], xu=min_max[1])

    def _evaluate(self, x, out, *args, **kwargs):

        # ----- PARAMETERS

        alpha = self.weight["alpha"]
        beta = self.weight["beta"]
        gamma = self.weight["gamma"]
        delta = self.weight["delta"]

        x_ml = self.encoder.from_genetic_to_ml(self.original, x).astype("float64")
        x_ml_mm = self.scaler.transform(x_ml)

        # f1 Maximize probability of target
        f1 = self.model.predict_proba(x_ml)[:, 1]
        f1[f1 < self.encoder.LOG_ALPHA] = self.encoder.LOG_ALPHA
        f1 = np.log(f1)

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self.original_mm[1:], axis=1)
        f2 = l2_distance

        # f3 Maximize amount (f3 don't need scaler if amount > 1)
        amount = copy.deepcopy(x_ml[:, 0])
        amount[amount <= 0] = self.encoder.AMOUNT_BETA
        f3 = 1 / amount

        # f4 Domain constraints

        constraints = venus_constraints.evaluate(x_ml)

        if self.scale_objectives:
            f1 = self.encoder.f1_scaler.transform(f1.reshape(-1, 1))[:, 0]
            f2 = self.encoder.f2_scaler.transform(f2.reshape(-1, 1))[:, 0]
            constraints = self.encoder.constraint_scaler.transform(constraints)

        out["F"] = alpha * f1 + beta * f2 + gamma * f3
        out["G"] = constraints * delta

        if self.save_history:
            self.history["f1"].append(f1)
            self.history["f2"].append(f2)
            self.history["f3"].append(f3)
            self.history["g1"].append(constraints)
