import abc
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
