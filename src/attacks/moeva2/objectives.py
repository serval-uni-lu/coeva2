import abc

import numpy as np


class Objectives(abc.ABC, metaclass=abc.ABCMeta):
    pass

    def evaluate(self, x, x_f, x_f_mm, x_ml) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_nb_objectives(self) -> int:
        raise NotImplementedError
