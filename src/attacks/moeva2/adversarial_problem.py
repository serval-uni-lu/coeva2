from pymoo.model.problem import Problem
import numpy as np

from .constraints import Constraints
from .feature_encoder import FeatureEncoder
from .classifier import Classifier

NB_OBJECTIVES = 3


class AdversarialProblem(Problem):
    def __init__(
        self,
        x_clean: np.ndarray,
        classifier: Classifier,
        y_clean: int,
        encoder: FeatureEncoder,
        constraints: Constraints,
        fun_distance_preprocess=lambda x: x,
        norm=None,
        save_history="none",
    ):
        # Parameters
        self.x_clean = x_clean
        self.classifier = classifier
        self.y_clean = y_clean
        self.encoder = encoder
        self.constraints = constraints
        self.fun_distance_preprocess = fun_distance_preprocess
        self.norm = norm
        self.save_history = save_history

        # Optional parameters
        self.save_history = save_history
        self.norm = norm

        # Caching
        self.x_clean_distance = self.fun_distance_preprocess(x_clean.reshape(1, -1))[0]

        # Computed attributes
        xl, xu = encoder.get_min_max_genetic()

        # Future storage
        self.history = []

        super().__init__(
            n_var=self.encoder.get_genetic_v_length(),
            n_obj=self.get_nb_objectives(),
            n_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_initial_state(self):
        return self.x_clean

    def get_history(self):
        return self.history

    def get_nb_objectives(self):
        return NB_OBJECTIVES

    def _obj_misclassify(self, x_ml: np.ndarray) -> np.ndarray:
        y_pred = self.classifier.predict_proba(x_ml)[:, self.y_clean]
        return y_pred

    def _obj_distance(self, x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:

        if self.norm in ["inf", np.inf]:
            distance = np.linalg.norm(x_1 - x_2, ord=np.inf, axis=1)
        elif self.norm in ["2", 2]:
            distance = np.linalg.norm(x_1 - x_2, ord=2, axis=1)
        else:
            raise NotImplementedError

        return distance

    def _calculate_constraints(self, x):
        G = self.constraints.evaluate(x)
        G = G * (G > 0).astype(np.float)

        return G

    def _evaluate(self, x, out, *args, **kwargs):

        # print("Evaluate")

        # Sanity check
        if (x - self.xl < 0).sum() > 0:
            print("Lower than lower bound.")

        if (x - self.xu > 0).sum() > 0:
            print("Lower than lower bound.")

        # --- Prepare necessary representation of the samples

        # Retrieve original representation

        x_adv = self.encoder.genetic_to_ml(x, self.x_clean)

        obj_misclassify = self._obj_misclassify(x_adv)

        obj_distance = self._obj_distance(
            self.fun_distance_preprocess(x_adv), self.x_clean_distance
        )

        all_constraints = self._calculate_constraints(x_adv)
        obj_constraints = all_constraints.sum(axis=1)

        F = [obj_misclassify, obj_distance, obj_constraints]

        # --- Output
        out["F"] = np.column_stack(F)

        # Save output
        if "reduced" in self.save_history:
            self.history.append(out["F"])
        elif "full" in self.save_history:
            self.history.append(np.concatenate((out["F"], all_constraints), axis=1))
