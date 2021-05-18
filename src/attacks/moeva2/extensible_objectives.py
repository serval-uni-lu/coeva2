###
#
#
# DO NOT USE
#
#
###
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.objectives import Objectives

AVOID_ZERO = 0.00000001


class ExtensibleObjectives(Objectives):
    def __init__(
        self,
        x_initial_state_f_mm,
        classifier,
        minimize_class,
        additional_objectives,
        scale_objectives=True,
    ):
        self._x_initial_state_f_mm = x_initial_state_f_mm
        self._classifier = classifier
        self._minimize_class = minimize_class
        self._additional_objectives = additional_objectives
        self._scale_objectives = scale_objectives

        # There are two default objectives.
        self._nb_objectives = 2 + len(additional_objectives)

        # Objective scalers (Compute only once)
        self._f1_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f1_scaler.fit([[np.log(AVOID_ZERO)], [np.log(1)]])

        self._f2_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f2_scaler.fit([[0], [np.sqrt(self._x_initial_state_f_mm.shape[0])]])

    def evaluate(self, x, x_f, x_f_mm, x_ml) -> np.ndarray:

        # Initial objectives
        F = [self._obj_misclassify(x_ml), self._obj_distance(x_f_mm)]

        # Additional objectives
        F = F + [
            obj_fun(x, x_f, x_f_mm, x_ml) for obj_fun in self._additional_objectives
        ]

        return np.column_stack(F)

    def _obj_misclassify(self, x_ml: np.ndarray) -> np.ndarray:
        f1 = self._classifier.predict_proba(x_ml)[:, self._minimize_class]
        f1[f1 < AVOID_ZERO] = AVOID_ZERO
        f1 = np.log(f1)

        if self._scale_objectives:
            f1 = self._f1_scaler.transform(f1.reshape(-1, 1))[:, 0]

        return f1

    def _obj_distance(self, x_mm: np.ndarray) -> np.ndarray:
        f2 = np.linalg.norm(x_mm[:, 1:] - self._x_initial_state_f_mm[1:], axis=1)
        if self._scale_objectives:
            f2 = self._f2_scaler.transform(f2.reshape(-1, 1))[:, 0]
        return f2

    def get_nb_objectives(self) -> int:
        return self._nb_objectives
