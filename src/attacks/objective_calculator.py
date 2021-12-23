import sys

import numpy
import numpy as np

from .moeva2.constraints.constraints import Constraints
from .moeva2.feature_encoder import get_encoder_from_constraints

numpy.set_printoptions(threshold=sys.maxsize)


def objectives_to_dict(objectives):
    objectives_dict = {}
    for objectives_i, objective in enumerate(objectives):
        objectives_dict[f"o{objectives_i+1}"] = objective
    return objectives_dict


class ObjectiveCalculator:
    def __init__(
        self,
        classifier,
        constraints: Constraints,
        thresholds: dict,
        fun_distance_preprocess=lambda x: x,
        norm=None,
        n_jobs=1,
    ):
        self._classifier = classifier
        self._constraints = constraints
        self.fun_distance_preprocess = fun_distance_preprocess
        self._thresholds = thresholds
        self._encoder = get_encoder_from_constraints(self._constraints)
        self.norm = norm
        self.n_jobs = n_jobs

    def _calc_fitness(self, x_clean, y_clean, x_adv):
        x_adv_c_score = self._constraints.evaluate(
            x_adv.reshape(-1, x_adv.shape[-1])
        ).reshape((*x_adv.shape[:-1], -1))

        y_score_filter = (
            np.arange(x_adv.shape[0] * x_adv.shape[1]),
            np.repeat(y_clean, x_adv.shape[1]),
        )

        x_adv_for_classifier = x_adv.reshape(-1, x_adv.shape[-1])

        x_adv_m_score = (
            self._classifier.predict_proba(x_adv.reshape(-1, x_adv.shape[-1]))
        )[y_score_filter].reshape((x_adv.shape[:-1]))

        delta = self.fun_distance_preprocess(
            x_adv.reshape(-1, x_adv.shape[-1])
        ) - self.fun_distance_preprocess(np.repeat(x_clean, x_adv.shape[1], axis=0))

        x_adv_distance_score = np.linalg.norm(delta, ord=self.norm, axis=1).reshape(
            (x_adv.shape[:-1])
        )
        return x_adv_c_score, x_adv_m_score, x_adv_distance_score

    def calculate_objectives(self, x_clean, y_clean, x_adv):
        x_adv_c_score, x_adv_m_score, x_adv_distance_score = self._calc_fitness(
            x_clean, y_clean, x_adv
        )

        x_adv_c = np.max(x_adv_c_score, axis=-1) <= 0
        x_adv_m = x_adv_m_score < self._thresholds["model"]
        x_adv_distance = x_adv_distance_score <= self._thresholds["distance"]

        return x_adv_c, x_adv_m, x_adv_distance

    def get_success_rates(self, x_clean, y_clean, x_adv):
        x_adv_c, x_adv_m, x_adv_distance = self.calculate_objectives(
            x_clean, y_clean, x_adv
        )
        objectives = [
            x_adv_c,
            x_adv_m,
            x_adv_distance,
            x_adv_c * x_adv_m,
            x_adv_c * x_adv_distance,
            x_adv_m * x_adv_distance,
            x_adv_c * x_adv_m * x_adv_distance,
        ]
        objectives = np.array([np.max(objective, axis=-1) for objective in objectives])
        success_rate = np.mean(objectives, axis=1)
        return success_rate
