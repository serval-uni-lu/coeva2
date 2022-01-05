import sys

import numpy
import numpy as np
from tqdm import tqdm

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

        x_adv_c_score = (
            1
            - self._constraints.check_constraints(
                np.repeat(x_clean, x_adv.shape[1], axis=0),
                x_adv.reshape(-1, x_adv.shape[-1]),
            ).reshape((*x_adv.shape[:-1], -1))
        )

        y_score_filter = (
            np.arange(x_adv.shape[0] * x_adv.shape[1]),
            np.repeat(y_clean, x_adv.shape[1]),
        )

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

    def _calc_obj_from_fitness(
        self, x_adv_c_score, x_adv_m_score, x_adv_distance_score
    ):
        x_adv_c = np.max(x_adv_c_score, axis=-1) <= 0
        x_adv_m = x_adv_m_score < self._thresholds["model"]
        x_adv_distance = x_adv_distance_score <= self._thresholds["distance"]

        return x_adv_c, x_adv_m, x_adv_distance

    def calculate_objectives(self, x_clean, y_clean, x_adv):
        x_adv_c_score, x_adv_m_score, x_adv_distance_score = self._calc_fitness(
            x_clean, y_clean, x_adv
        )

        return self._calc_obj_from_fitness(
            x_adv_c_score, x_adv_m_score, x_adv_distance_score
        )

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

    def get_successful_attacks(
        self,
        x_clean,
        y_clean,
        x_adv,
        preferred_metrics="misclassification",
        order="asc",
        max_inputs=-1,
        return_index_success=False,
    ):
        x_adv_c_score, x_adv_m_score, x_adv_distance_score = self._calc_fitness(
            x_clean, y_clean, x_adv
        )

        x_adv_c, x_adv_m, x_adv_distance = self._calc_obj_from_fitness(
            x_adv_c_score, x_adv_m_score, x_adv_distance_score
        )
        x_adv_cmd = x_adv_c * x_adv_m * x_adv_distance

        successful_attacks = []
        successful_index = []
        for x_i, x in tqdm(enumerate(x_clean)):

            index_success = np.where(x_adv_cmd[x_i])[0]
            if len(index_success) > 0:
                successful_index.append(x_i)
                x_success = x_adv[x_i][index_success]
                if preferred_metrics == "misclassification":
                    x_success_score = x_adv_m_score[x_i][index_success]
                elif preferred_metrics == "distance":
                    x_success_score = x_adv_distance_score[x_i][index_success]
                else:
                    raise NotImplementedError

                sort_i = np.argsort(x_success_score)
                if order != "asc":
                    sort_i = sort_i[::-1]

                x_success = x_success[sort_i]
                if max_inputs >= 0:
                    x_success = x_success[:max_inputs]
                successful_attacks.append(x_success)
            else:
                successful_attacks.append([])

        if return_index_success:
            return successful_attacks, np.array(successful_index)
        else:
            return successful_attacks
