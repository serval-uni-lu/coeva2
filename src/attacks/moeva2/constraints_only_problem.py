from pymoo.model.problem import Problem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .constraints import Constraints
from .feature_encoder import FeatureEncoder
from .classifier import Classifier

AVOID_ZERO = 0.00000001
NB_OBJECTIVES = 1


class ConstraintsOnlyProblem(Problem):
    # Some argument are not used but respect the contract (i.e. attacks sends correct parameters)
    def __init__(
        self,
        x_initial_state: np.ndarray,
        classifier: Classifier,
        minimize_class: int,
        encoder: FeatureEncoder,
        constraints: Constraints,
        scale_objectives: True,
        save_history=False,
        ml_scaler=None,
    ):
        # Essential passed parameters
        self.x_initial_ml = x_initial_state
        self._constraints = constraints
        self._encoder = encoder

        # Optional parameters
        self._save_history = save_history

        # Computed attributes
        self._x_initial_f_mm = encoder.normalise(x_initial_state)
        xl, xu = encoder.get_min_max_genetic()

        self._history = []
        super().__init__(
            n_var=self._encoder.get_genetic_v_length(),
            n_obj=self.get_nb_objectives(),
            n_constr=constraints.get_nb_constraints(),
            xl=xl,
            xu=xu,
        )

    def get_initial_state(self):
        return self.x_initial_ml

    def get_history(self):
        return self._history

    def get_nb_objectives(self):
        return NB_OBJECTIVES

    def _evaluate(self, x, out, *args, **kwargs):

        # --- Prepare necessary representation of the samples

        # Genetic representation is in x

        # Machine learning representation
        x_f = self._encoder.genetic_to_ml(x, self.x_initial_ml)

        # Min max scaled representation
        x_f_mm = self._encoder.normalise(x_f)

        # --- Domain constraints
        G = self._constraints.evaluate(x_f)

        # --- Out and History
        out["F"] = np.zeros(x.shape[0])
        out["G"] = G

        if self._save_history:
            self._history.append(out)
