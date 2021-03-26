import pickle
from typing import List

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.constraints import Constraints
import numpy as np

from attacks.coeva2.feature_encoder import get_encoder_from_constraints
from attacks.coeva2.result_process import EfficientResult
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from utils import in_out
config = in_out.get_parameters()


class ObjectiveCalculator:
    def __init__(
        self,
        classifier: Classifier,
        constraints: Constraints,
        threshold: float,
        #high_amount: int,
        #amount_index: int,
    ):
        self._classifier = classifier
        self._constraints = constraints
        self._encoder = get_encoder_from_constraints(constraints)
        self._threshold = threshold
        #self._high_amount = high_amount
        #self._amount_index = amount_index
        if config["paths"]["ml_scaler"] is not None:
            with open(config["paths"]["ml_scaler"], 'rb') as f:
                self._ml_scaler = pickle.load(f)
        else:
            self._ml_scaler = None

    def _objective_per_individual(self, result: EfficientResult) -> np.ndarray:
        x = np.array(list(map(lambda e: e.X, result.pop))).astype(np.float64)
        x_ml = self._encoder.genetic_to_ml(x, result.initial_state)

        respectsConstraints = (
            (self._constraints.evaluate(x_ml)).sum(axis=1)
            <= 0
        ).astype(np.int64)
        # print(self._constraints.evaluate(x_ml))

        if self._ml_scaler is not None:
            f1 = self._classifier.predict_proba(self._ml_scaler.transform(x_ml))
            f1 = np.array(f1).flatten()
        else:
            f1 = self._classifier.predict_proba(x_ml)[:, 1]

        # f1 = self._classifier.predict_proba(x_ml)
        # f1 = np.array(f1).flatten()
        isMisclassified = np.array(f1 < self._threshold).astype(np.int64)

        #isHighAmount = (x_ml[:, self._amount_index] >= self._high_amount).astype(
            #np.int64
        #)

        o1 = respectsConstraints
        o2 = isMisclassified
        o3 = o1 * o2
        #o4 = o3 * isHighAmount
        return np.array([respectsConstraints, isMisclassified, o3])

    def _objective_per_initial_sample(self, result: EfficientResult):
        objectives = self._objective_per_individual(result)
        objectives = objectives.sum(axis=1)
        objectives = (objectives > 0).astype(np.int64)
        return objectives

    def success_rate(self, results: List[EfficientResult]):
        objectives = np.array(
            [self._objective_per_initial_sample(result) for result in results]
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
