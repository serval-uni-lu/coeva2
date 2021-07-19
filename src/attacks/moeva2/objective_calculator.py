from typing import List, Dict, Union
from pymoo.model.problem import Problem

import numpy as np

from .feature_encoder import get_encoder_from_constraints
from .result_process import EfficientResult
import sys
import numpy
from tqdm import tqdm
from .classifier import Classifier
from .constraints import Constraints

numpy.set_printoptions(threshold=sys.maxsize)


class ObjectiveCalculator:
    def __init__(
        self,
        classifier: Classifier,
        constraints: Constraints,
        minimize_class: int,
        thresholds: dict,
        min_max_scaler,
        norm=np.inf,
        ml_scaler=None,
        problem_class=None,
    ):
        self._classifier = classifier
        self._constraints = constraints
        self._thresholds = thresholds
        self._ml_scaler = ml_scaler
        self._problem_class = problem_class
        self._minimize_class = minimize_class
        self._encoder = get_encoder_from_constraints(self._constraints)
        self._min_max_scaler = min_max_scaler
        self.norm = norm

    def _calculate_objective(self, x_initial, x_f):

        # Constraints
        constraint_violation = Problem.calc_constraint_violation(
            self._constraints.evaluate(x_f)
        ).reshape(-1)

        # Misclassify

        x_ml = x_f
        if self._ml_scaler is not None:
            x_ml = self._ml_scaler.transform(x_f)
        f1 = self._classifier.predict_proba(x_ml)[:, self._minimize_class]

        # Distance

        # Scale and check scaling

        x_i_scaled = self._min_max_scaler.transform(x_initial.reshape(1, -1))
        x_scaled = self._min_max_scaler.transform(x_f)
        tol = 0.0001
        assert np.all(x_i_scaled >= 0 - tol)
        assert np.all(x_i_scaled <= 1 + tol)
        assert np.all(x_scaled >= 0 - tol)
        assert np.all(x_scaled <= 1 + tol)

        f2 = np.linalg.norm(
            x_i_scaled - x_scaled,
            ord=self.norm,
            axis=1,
        )

        return np.column_stack([constraint_violation, f1, f2])

    def _objective_respected(self, objective_values):
        constraints_respected = objective_values[:, 0] <= 0
        misclassified = objective_values[:, 1] < self._thresholds["f1"]
        l2_in_ball = objective_values[:, 2] <= self._thresholds["f2"]
        return np.column_stack(
            [
                constraints_respected,
                misclassified,
                l2_in_ball,
                constraints_respected * misclassified,
                constraints_respected * l2_in_ball,
                misclassified * l2_in_ball,
                constraints_respected * misclassified * l2_in_ball,
            ]
        )

    def _objective_array(self, x_initial, x_f):
        objective_values = self._calculate_objective(x_initial, x_f)
        return self._objective_respected(objective_values)

    def success_rate(self, x_initial, x_f):
        return self._objective_array(x_initial, x_f).mean(axis=0)

    def at_least_one(self, x_initial, x_f):
        return np.array(self.success_rate(x_initial, x_f) > 0)

    def success_rate_3d(self, x_initial, x):
        at_leat_one = np.array(
            [
                self.success_rate(x_initial[i], e) > 0
                for i, e in tqdm(enumerate(x), total=len(x))
            ]
        )
        return at_leat_one.mean(axis=0)

    def success_rate_genetic(self, results: List[EfficientResult]):

        initial_states = [result.initial_state for result in results]
        # Use last pop or all gen pareto front to compute objectives.
        # pops_x = [result.X.astype(np.float64) for result in results]
        # pops_x = [result.pareto.astype(np.float64) for result in results]
        pops_x = [
            np.array([ind.X.astype(np.float64) for ind in result.pop])
            for result in results
        ]
        # Convert to ML representation
        pops_x_f = [
            self._encoder.genetic_to_ml(pops_x[i], initial_states[i])
            for i in range(len(results))
        ]

        return self.success_rate_3d(initial_states, pops_x_f)

    def get_success(self, x_initial, x_f):
        raise NotImplementedError

    def get_success_full_inputs(
        self, prefered_metrics="distance", order="asc", max_inputs=-1
    ):

        at_leat_one = np.array(
            [
                self.success_rate(x_initial[i], e) > 0
                for i, e in tqdm(enumerate(x), total=len(x))
            ]
        )
        return at_leat_one.mean(axis=0)
