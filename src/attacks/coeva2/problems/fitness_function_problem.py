import copy

from pymoo.model.problem import Problem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .. import venus_constraints
from ..classifier import Classifier
from ..constraints import Constraints
from ..feature_encoder import FeatureEncoder

AVOID_ZERO = 0.00000001


class FitnessFunctionProblem(Problem):



    def __init__(self, x_initial_state: np.ndarray, classifier: Classifier,
                 encoder: FeatureEncoder, constraints: Constraints, scale_objectives=True, save_history=False):
        self._x_initial_ml = x_initial_state
        self._x_initial_ml_mm = encoder.normalise(x_initial_state)
        self._x_initial_ga = encoder.ml_to_genetic(x_initial_state)
        self._classifier = classifier
        self._constraints = constraints
        self._scale_objectives = scale_objectives
        self._save_history = save_history
        self._encoder = encoder
        xl, xu = encoder.get_min_max_genetic()
        self._history = {
            "f1": [],
            "f2": [],
            "f3": [],
            "g1": [],
        }
        self._init_objective_scaler()
        super().__init__(n_var=len(self._x_initial_ga.shape[0]), n_obj=1, n_constr=constraints.get_nb_constraints(),
                         xl=xl, xu=xu)

    def _init_objective_scaler(self):
        self._f1_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f1_scaler.fit([[np.log(AVOID_ZERO)], [np.log(1)]])
        self._f2_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f2_scaler.fit([[0], [np.sqrt(self._x_initial_ml.shape[0])]])

    def _evaluate(self, x, out, *args, **kwargs):

        x_ml = self._encoder.genetic_to_ml(x)
        x_ml_mm = self._encoder.normalise(x_ml)

        # f1 Maximize probability of target
        f1 = self._classifier.predict_proba(x_ml)[:, 1]
        f1[f1 < AVOID_ZERO] = AVOID_ZERO
        f1 = np.log(f1)

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self._x_initial_ml_mm[1:], axis=1)
        f2 = l2_distance

        # f3 Maximize amount (f3 don't need scaler if amount > 1)
        amount = copy.deepcopy(x_ml[:, 0])
        amount[amount <= 0] = AVOID_ZERO
        f3 = 1 / amount

        # f4 Domain constraints

        constraints = self._constraints.evaluate(x_ml)

        if self._scale_objectives:
            f1 = self._f1_scaler.transform(f1.reshape(-1, 1))[:, 0]
            f2 = self._f2_scaler.transform(f2.reshape(-1, 1))[:, 0]
            constraints = self.encoder.constraint_scaler.transform(constraints)

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = constraints
