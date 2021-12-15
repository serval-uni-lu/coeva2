import abc

import numpy as np


class Dataset(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_train_test(self) -> (np.ndarray, np.ndarray, np.array, np.ndarray):
        raise NotImplementedError
