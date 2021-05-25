from pymoo.model.problem import Problem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .constraints import Constraints
from .feature_encoder import FeatureEncoder
from .classifier import Classifier

AVOID_ZERO = 0.00000001
NB_OBJECTIVES = 3


class DefaultProblem(Problem):
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
        self.classifier = classifier
        self.minimize_class = minimize_class
        self._constraints = constraints
        self.encoder = encoder

        # Optional parameters
        self.scale_objectives = scale_objectives
        self._save_history = save_history

        # Computed attributes
        self.x_initial_f_mm = encoder.normalise(x_initial_state)
        self._create_default_scaler()
        xl, xu = encoder.get_min_max_genetic()

        self._ml_scaler = ml_scaler

        self._history = []

        super().__init__(
            n_var=self.encoder.get_genetic_v_length(),
            n_obj=self.get_nb_objectives(),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_initial_state(self):
        return self.x_initial_ml

    def get_history(self):
        return self._history

    def get_nb_objectives(self):
        return NB_OBJECTIVES

    def _create_default_scaler(self):
        # Objective scalers (Compute only once)
        self._f1_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f1_scaler.fit([[np.log(AVOID_ZERO)], [np.log(1)]])

        self._f2_scaler = MinMaxScaler(feature_range=(0, 1))
        self._f2_scaler.fit([[0], [np.sqrt(self.x_initial_f_mm.shape[0])]])

    def _obj_misclassify(self, x_ml: np.ndarray) -> np.ndarray:
        f1 = self.classifier.predict_proba(x_ml)[:, self.minimize_class]
        # print(np.mean(f1))
        f1[f1 < AVOID_ZERO] = AVOID_ZERO
        f1 = np.log(f1)

        if self.scale_objectives:
            f1 = self._f1_scaler.transform(f1.reshape(-1, 1))[:, 0]

        return f1

    def _obj_distance_l0(self, x_f: np.ndarray) -> np.ndarray:

        f2 = np.abs(x_f - self.x_initial_ml)
        f2 = np.count_nonzero(f2 > 0.001, axis=1)
        # print(f2.min())
        if self.scale_objectives:
            f2 = f2/self.x_initial_f_mm.shape[0]
        return f2

    def _obj_distance(self, x_f_mm: np.ndarray) -> np.ndarray:

        f2 = np.linalg.norm(x_f_mm - self.x_initial_f_mm, axis=1)
        # print(np.mean(f2))
        if self.scale_objectives:
            f2 = self._f2_scaler.transform(f2.reshape(-1, 1))[:, 0]
        return f2

    def _evaluate(self, x, out, *args, **kwargs):

        # --- Prepare necessary representation of the samples

        # Genetic representation
        x = x

        # Machine learning representation
        x_f = self.encoder.genetic_to_ml(x, self.x_initial_ml)

        # Min max scaled representation
        x_f_mm = self.encoder.normalise(x_f)

        # ML scaled
        x_ml = x_f
        if self._ml_scaler is not None:
            x_ml = self._ml_scaler.transform(x_f)

        # --- Objectives
        f1 = self._obj_misclassify(x_ml)
        f2 = self._obj_distance(x_f_mm)

        # F = [f1, f2] + self._evaluate_additional_objectives(x, x_f, x_f_mm, x_ml)

        # --- Domain constraints
        G = self._constraints.evaluate(x_f)
        if self.scale_objectives:
            G = self._constraints.normalise(G)

        CV = Problem.calc_constraint_violation(
            G
        ).reshape(-1)

        if self.scale_objectives:
            CV = CV / G.shape[1]

        # print(CV.min())

        F = [f1, f2, CV] + self._evaluate_additional_objectives(x, x_f, x_f_mm, x_ml)

        # --- Out and History
        out["F"] = np.column_stack(F)
        # out["G"] = G

        if self._save_history:
            self._history.append(out)

    def _evaluate_additional_objectives(self, x, x_f, x_f_mm, x_ml):
        return []
