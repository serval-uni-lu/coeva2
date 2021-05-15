import abc
import logging
from abc import ABC
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.attacks.coeva2.constraints import Constraints


class FileConstraints(Constraints, ABC):

    def __init__(self, feature_path: str, constraints_path: str):
        self._provision_constraints_min_max(constraints_path)
        self._provision_feature_constraints(feature_path)
        self._fit_scaler()

    def _fit_scaler(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        min_c, max_c = self.get_constraints_min_max()
        self._scaler.fit([min_c, max_c])

    def _provision_feature_constraints(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._feature_min = df["min"].to_numpy()
        self._feature_max = df["max"].to_numpy()
        self._mutable_mask = df["mutable"].to_numpy()
        self._feature_type = df["type"].to_numpy()

    def _provision_constraints_min_max(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._constraints_min = df["min"].to_numpy()
        self._constraints_max = df["max"].to_numpy()
        self._fit_scaler()

    def get_nb_constraints(self) -> int:
        return self._constraints_min.shape[0]

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.transform(x)

    def get_constraints_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._constraints_min, self._constraints_max

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def get_feature_min_max(self, dynamic_input=None) -> Tuple[np.ndarray, np.ndarray]:

        # By default min and max are the extreme values
        feature_min = np.array([np.finfo(np.float32).min] * self._feature_min.shape[0])
        feature_max = np.array([np.finfo(np.float32).max] * self._feature_max.shape[0])

        # Creating the mask of value that should be provided by input
        min_dynamic = self._feature_min.astype(str) == "dynamic"
        max_dynamic = self._feature_max.astype(str) == "dynamic"

        # Replace de non dynamic value by the value provided in the definition
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
