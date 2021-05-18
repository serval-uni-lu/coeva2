###
#
#
# DO NOT USE - Old version
#
#
###


import copy
import pickle

from pymoo.model.problem import Problem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .classifier import Classifier
from .constraints import Constraints
from .feature_encoder import FeatureEncoder

from utils import in_out

from .objectives import Objectives

config = in_out.get_parameters()

AVOID_ZERO = 0.00000001


class Moeva2Problem(Problem):
    def __init__(
        self,
        x_initial_state: np.ndarray,
        classifier: Classifier,
        encoder: FeatureEncoder,
        constraints: Constraints,
        additional_objectives: Objectives,
        save_history=False,
        ml_scaler=None,
    ):
        # Essential passed parameters
        self._x_initial_ml = x_initial_state
        self._classifier = classifier,
        self._constraints = constraints
        self._additional_objectives = additional_objectives
        self._encoder = encoder

        # Optional parameters
        self._save_history = save_history

        # Computed attributes
        self._x_initial_f_mm = encoder.normalise(x_initial_state)
        xl, xu = encoder.get_min_max_genetic()

        if ml_scaler is not None:
            self._ml_scaler = ml_scaler

        self._history = []
        self._default_objectives = Def

        super().__init__(
            n_var=self._encoder.get_genetic_v_length(),
            n_obj=objectives.get_nb_objectives(),
            n_constr=constraints.get_nb_constraints(),
            xl=xl,
            xu=xu,
        )

    def get_initial_state(self):
        return self._x_initial_ml

    def get_history(self):
        return self._history

    def _evaluate(self, x, out, *args, **kwargs):

        # --- Prepare necessary representation of the samples

        # Genetic representation
        x = x

        # Machine learning representation
        x_f = self._encoder.genetic_to_ml(x, self._x_initial_ml)

        # Min max scaled representation
        x_f_mm = self._encoder.normalise(x_f)

        # ML scaled
        x_ml = x_f
        if self._ml_scaler is not None:
            x_ml = self._ml_scaler.transform(x_f)

        # --- Objectives
        F = self._objectives.evaluate(x, x_f, x_f_mm, x_ml)

        # --- Domain constraints
        G = self._constraints.evaluate(x_f)

        # --- Out and History
        out["F"] = F
        out["G"] = G

        if self._save_history:
            self._history.append(out)
