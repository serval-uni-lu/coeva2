import abc
from typing import Tuple

import numpy as np


class Constraints(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the distance to constraints satisfaction of x. This method should be overridden by the attacker.

        Args:
            x (np.ndarray): An array of shape (n_samples, n_features) containing the sample to evaluate.

        Returns:
            np.ndarray: An array of shape (n_samples, n_constraints) representing the distance to constraints
            satisfaction of each sample.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_nb_constraints(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def normalise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_constraints_min_max(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_mutable_mask(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_type(self) -> np.ndarray:
        raise NotImplementedError

    def check_constraints_error(self, x: np.ndarray):
        constraints = self.evaluate(x)
        constraints_violated = (constraints > 0).sum()
        if constraints_violated > 0:
            raise ValueError(f"Constraints not respected {constraints_violated} times.")
