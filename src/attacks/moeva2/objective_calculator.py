from typing import List, Dict
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
# from utils import in_out

# config = in_out.get_parameters()


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

    def _objective_array(self, x_initial, x_f):

        # Constraints
        constraint_violation = Problem.calc_constraint_violation(
            self._constraints.evaluate(x_f)
        ).reshape(-1)
        constraints_respected = constraint_violation <= 0

        # Classifier

        x_ml = x_f
        if self._ml_scaler is not None:
            x_ml = self._ml_scaler.transform(x_f)

        f1 = self._classifier.predict_proba(x_ml)[:, self._minimize_class]
        misclassified = f1 < self._thresholds["f1"]

        # In the ball

        x_i_scaled = self._min_max_scaler.transform(x_initial.reshape(1, -1))
        x_scaled = self._min_max_scaler.transform(x_f)
        tol = 0.0001
        assert np.all(x_i_scaled >= 0 - tol)
        assert np.all(x_i_scaled <= 1 + tol)
        assert np.all(x_scaled >= 0 - tol)
        assert np.all(x_scaled <= 1 + tol)

        l2 = np.linalg.norm(
            x_i_scaled - x_scaled,
            ord=self.norm,
            axis=1,
        )
        # print(np.min(l2[(constraints_respected * misclassified)]))

        l2_in_ball = l2 <= self._thresholds["f2"]
        # print(l2)
        # print(np.max(l2))
        # Additional
        # to implement

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

    def success_rate_bis(self, x_initial, x_f):
        return self._objective_array(x_initial, x_f).mean(axis=0)

    def at_least_one(self, x_initial, x_f):
        return np.array(self.success_rate_bis(x_initial, x_f) > 0)

    def success_rate_genetic(self, results: List[EfficientResult]):

        initial_states = [result.initial_state for result in results]
        # pops_x = [
        #     result.X.astype(np.float64) for result in results
        # ]
        pops_x = [
            result.pareto.astype(np.float64) for result in results
        ]
        pops_x_f = [
            self._encoder.genetic_to_ml(pops_x[i], initial_states[i])
            for i in range(len(results))
        ]

        # For each pop (results) check if the success rate is over 1
        pops_at_least_one = np.array(
            [
                self.success_rate_bis(initial_states[i], pop_x_f) > 0
                for i, pop_x_f in tqdm(enumerate(pops_x_f), total=len(pops_x_f))
            ]
        )

        return pops_at_least_one.mean(axis=0)

    def success_rate_3d(self, x_initial, x):
        at_leat_one = np.array(
            [self.success_rate_bis(x_initial[i], e) > 0 for i, e in tqdm(enumerate(x), total=len(x))]
        )
        return at_leat_one.mean(axis=0)

    def _objective_per_individual(self, result: EfficientResult) -> np.ndarray:
        x = np.array(list(map(lambda e: e.X, result.X))).astype(np.float64)
        x_ml = self._encoder.genetic_to_ml(x, result.initial_state)

        respectsConstraints = (
            (self._constraints.evaluate(x_ml)).sum(axis=1) <= 0
        ).astype(np.int64)
        # print(self._constraints.evaluate(x_ml))

        if self._ml_scaler is not None:
            f1 = self._classifier.predict_proba(self._ml_scaler.transform(x_ml))
            f1 = np.array(f1).flatten()
        else:
            f1 = self._classifier.predict_proba(x_ml)[:, 1]

        # f1 = self._classifier.predict_proba(x_ml)
        # f1 = np.array(f1).flatten()
        isMisclassified = np.array(f1 < self._thresholds["f1"]).astype(np.int64)

        # isHighAmount = (x_ml[:, self._amount_index] >= self._high_amount).astype(
        # np.int64
        # )

        o1 = respectsConstraints
        o2 = isMisclassified
        o3 = o1 * o2
        # o4 = o3 * isHighAmount
        return np.array([respectsConstraints, isMisclassified, o3])

    def _objective_per_initial_sample(self, result: EfficientResult):
        objectives = self._objective_per_individual(result)
        objectives = objectives.sum(axis=1)
        objectives = (objectives > 0).astype(np.int64)
        return objectives

    def success_rate(self, results: List[EfficientResult]):
        objectives = np.array(
            [self._objective_per_initial_sample(result) for result in tqdm(results)]
        )
        success_rates = np.apply_along_axis(
            lambda x: x.sum() / x.shape[0], 0, objectives
        )
        return success_rates

    def get_successful_attacks(self, results: List[EfficientResult]) -> np.ndarray:
        training = []

        for result in results:
            adv_filter = self._objective_per_individual(result)[2].astype(np.bool)
            x = np.array(list(map(lambda e: e.X, result.pop))).astype(np.float64)
            x_ml = self._encoder.genetic_to_ml(x, result.initial_state)
            training.append(x_ml[adv_filter])

        return np.concatenate(training)

    def get_generated(
        self, results: List[EfficientResult]
    ) -> List[Dict[str, np.ndarray]]:
        generated = []

        for result in results:
            adv_filter = self._objective_per_individual(result)[2].astype(np.bool)
            x = np.array(list(map(lambda e: e.X, result.pop))).astype(np.float64)
            x_ml = self._encoder.genetic_to_ml(x, result.initial_state)
            x_adv_success = x_ml[adv_filter]
            x_adv_failure = x_ml[~adv_filter]
            generated.append(
                {
                    "x_initial_state": result.initial_state,
                    "x_adv_success": x_adv_success,
                    "x_adv_fail": x_adv_failure,
                }
            )

        return generated

    def get_single_successful_attack(
        self, results: List[EfficientResult]
    ) -> np.ndarray:
        training = []
        for result in results:
            adv_filter = self._objective_per_individual(result)[2].astype(np.bool)
            x = np.array(list(map(lambda e: e.X, result.pop))).astype(np.float64)
            x_ml = self._encoder.genetic_to_ml(x, result.initial_state)
            successful = x_ml[adv_filter]
            if successful.shape[0] > 0:
                rnd_idx = np.random.choice(successful.shape[0], size=1, replace=False)
                training.append(successful[rnd_idx, :])
        return np.concatenate(training)
