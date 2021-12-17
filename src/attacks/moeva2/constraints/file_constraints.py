import abc
from itertools import combinations
from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.constraints.constraints import Constraints
import autograd.numpy as anp
import pandas as pd
import pickle
import logging
import pandas as pd

from src.examples.utils import constraints_augmented_np, constraints_augmented_tf


class FileConstraints(Constraints, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fix_features_types(self, x) -> Union[np.ndarray, tf.Tensor]:
        raise NotImplementedError

    def __init__(
        self,
        features_path: str,
    ):
        features = pd.read_csv(features_path)

        self._feature_type = features["type"].to_numpy()
        self._mutable_mask = features["mutable"].to_numpy()
        self._feature_min = features["min"].to_numpy()
        self._feature_max = features["max"].to_numpy()

    def normalise(self, x: np.ndarray) -> np.ndarray:
        None
        raise NotImplementedError

    def get_constraints_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        None
        raise NotImplementedError

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def get_feature_min_max(self, dynamic_input=None) -> Tuple[np.ndarray, np.ndarray]:

        # By default min and max are the extreme values
        feature_min = np.array([0.0] * self._feature_min.shape[0])
        feature_max = np.array([0.0] * self._feature_max.shape[0])

        # Creating the mask of value that should be provided by input
        min_dynamic = self._feature_min.astype(str) == "dynamic"
        max_dynamic = self._feature_max.astype(str) == "dynamic"

        # Replace de non-dynamic value by the value provided in the definition
        feature_min[~min_dynamic] = self._feature_min[~min_dynamic]
        feature_max[~max_dynamic] = self._feature_max[~max_dynamic]

        # If the dynamic input was provided, replace value for output, else do nothing (keep the extreme values)
        if dynamic_input is not None:
            feature_min[min_dynamic] = dynamic_input[min_dynamic]
            feature_max[max_dynamic] = dynamic_input[max_dynamic]

        # Raise warning if dynamic input waited but not provided
        dynamic_number = min_dynamic.sum() + max_dynamic.sum()
        if dynamic_number > 0 and dynamic_input is None:
            logging.getLogger().warning(
                f"{dynamic_number} feature min and max are dynamic but no input were provided."
            )

        return feature_min, feature_max

    def get_feature_type(self) -> np.ndarray:
        return self._feature_type
