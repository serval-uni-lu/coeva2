from math import comb
from typing import Union, Tuple

import numpy as np
import tensorflow as tf

from src.attacks.moeva2.constraints.constraints import Constraints
from src.constraints.utils import constraints_augmented_np
from src.experiments.united.important_utils import augment_data


class AugmentedConstraints(Constraints):
    def __init__(self, constraints0: Constraints, important_features):
        super().__init__()
        self.constraints0 = constraints0
        self.important_features = important_features
        self.nb_new_features = comb(self.important_features.shape[0], 2)

    def evaluate(self, x: np.ndarray, use_tensors: bool = False) -> np.ndarray:
        constraints_evaluation = self.constraints0.evaluate(x, use_tensors)

        # If not using tensor do nothing
        if not use_tensors:
            augmented_constraints = np.column_stack(
                constraints_augmented_np(
                    x, self.important_features[:, 0], self.important_features[:, 1]
                )
            )
            augmented_constraints[
                augmented_constraints <= self.constraints0.tolerance
            ] = 0.0
            constraints_evaluation = np.concatenate(
                [constraints_evaluation, augmented_constraints], axis=1
            )

        return constraints_evaluation

    def get_nb_constraints(self) -> int:
        return self.constraints0.get_nb_constraints() + self.nb_new_features

    def get_mutable_mask(self) -> np.ndarray:
        return np.concatenate(
            [
                self.constraints0.get_mutable_mask(),
                np.ones(self.nb_new_features).astype(np.bool),
            ]
        )

    def get_feature_min_max(self, dynamic_input=None) -> Tuple[np.ndarray, np.ndarray]:
        xl, xu = self.constraints0.get_feature_min_max(dynamic_input)
        xl = np.concatenate([xl, np.zeros(self.nb_new_features)])
        xu = np.concatenate([xu, np.ones(self.nb_new_features)])
        return xl, xu

    def fix_features_types(self, x) -> Union[np.ndarray, tf.Tensor]:
        x = self.constraints0.fix_features_types(x)
        new_tensor_v = tf.Variable(x)
        new_tensor_v = new_tensor_v.numpy()
        combi = -self.nb_new_features
        new_tensor_v = new_tensor_v[..., :combi]
        new_tensor_v = augment_data(new_tensor_v, self.important_features)
        return tf.convert_to_tensor(new_tensor_v, dtype=tf.float32)

    def get_feature_type(self) -> np.ndarray:
        return np.concatenate(
            [
                self.constraints0.get_feature_type(),
                np.repeat(["int"], self.nb_new_features),
            ]
        )
