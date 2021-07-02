from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.constraints import Constraints
import autograd.numpy as anp
import pandas as pd
import pickle
import logging


class BotnetConstraints(Constraints):
    def __init__(
        self,
        # amount_feature_index: int,
        feature_path: str,
        constraints_path: str,
    ):
        self._provision_constraints_min_max(constraints_path)
        self._provision_feature_constraints(feature_path)
        self._fit_scaler()
        # self._amount_feature_index = amount_feature_index

    def _fit_scaler(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        min_c, max_c = self.get_constraints_min_max()
        self._scaler = self._scaler.fit([min_c, max_c])

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # ----- PARAMETERS

        tol = 1e-3
        # should write a function in utils for this part
        with open("./data/botnet/feat_idx.pickle", "rb") as f:
            feat_idx = pickle.load(f)

        sum_idx = [0, 3, 6, 12, 15, 18]
        max_idx = [1, 4, 7, 13, 16, 19]
        min_idx = [2, 5, 8, 14, 17, 20]

        g1 = np.absolute(
            (
                    x[:, feat_idx["icmp_sum_s_idx"]].sum(axis=1)
                    + x[:, feat_idx["udp_sum_s_idx"]].sum(axis=1)
                    + x[:, feat_idx["tcp_sum_s_idx"]].sum(axis=1)
            )
            - (
                    x[:, feat_idx["bytes_in_sum_s_idx"]].sum(axis=1)
                    + x[:, feat_idx["bytes_out_sum_s_idx"]].sum(axis=1)
            )
        )
        g2 = np.absolute(
            (
                x[:, feat_idx["icmp_sum_d_idx"]].sum(axis=1)
                + x[:, feat_idx["udp_sum_d_idx"]].sum(axis=1)
                + x[:, feat_idx["tcp_sum_d_idx"]].sum(axis=1)
            )
            - (
                x[:, feat_idx["bytes_in_sum_d_idx"]].sum(axis=1)
                + x[:, feat_idx["bytes_out_sum_d_idx"]].sum(axis=1)
            )
        )

        constraints = [g1, g2]

        cons_idx = 3
        cons_idx, constraints0 = self.define_individual_constraints_pkts_bytes(
            x, cons_idx, feat_idx
        )
        constraints.extend(constraints0)
        cons_idx, constraints1 = self.define_individual_constraints(
            x, cons_idx, feat_idx, sum_idx, max_idx
        )
        constraints.extend(constraints1)
        cons_idx, constraints2 = self.define_individual_constraints(
            x, cons_idx, feat_idx, sum_idx, min_idx
        )
        constraints.extend(constraints2)
        cons_idx, constraints3 = self.define_individual_constraints(
            x, cons_idx, feat_idx, max_idx, min_idx
        )
        constraints.extend(constraints3)

        constraints = anp.column_stack(constraints)
        constraints[constraints <= tol] = 0.0

        return constraints

    # --------
    # PLEASE UPDATE THE NUMBER HERE
    # -------
    def get_nb_constraints(self) -> int:
        return 360

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return self._scaler.transform(x)

    def get_constraints_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._constraints_min, self._constraints_max

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    def get_feature_min_max(self, dynamic_input=None) -> Tuple[np.ndarray, np.ndarray]:

        # By default min and max are the extreme values
        feature_min = np.array([np.finfo(np.float).min] * self._feature_min.shape[0])
        feature_max = np.array([np.finfo(np.float).max] * self._feature_max.shape[0])

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

    @staticmethod
    def define_individual_constraints(x, cons_idx, feat_idx, upper_idx, lower_idx):
        constraints_part = []
        keys = list(feat_idx.keys())

        for i in range(len(upper_idx)):
            key = keys[upper_idx[i]]
            type_lower = keys[lower_idx[i]]
            type_upper = keys[upper_idx[i]]
            for j in range(len(feat_idx[key])):
                port_idx_lower = feat_idx[type_lower][j]
                port_idx_upper = feat_idx[type_upper][j]
                globals()["g%s" % cons_idx] = (
                    x[:, port_idx_lower] - x[:, port_idx_upper]
                )
                constraints_part.append(globals()["g%s" % cons_idx])
                cons_idx += 1
        return cons_idx, constraints_part

    @staticmethod
    def define_individual_constraints_pkts_bytes(x, cons_idx, feat_idx):
        constraints_part = []
        keys = list(feat_idx.keys())
        bytes_out = ["bytes_out_sum_s_idx", "bytes_out_sum_d_idx"]
        pkts_out = ["pkts_out_sum_s_idx", "pkts_out_sum_d_idx"]
        for i in range(len(bytes_out)):
            pkts = feat_idx[pkts_out[i]]
            bytes_ = feat_idx[bytes_out[i]]
            for j in range(len(bytes_out[i]) - 2):
                port_idx_pkts = pkts[j]
                port_idx_bytes = bytes_[j]
                a = x[:, port_idx_bytes]
                b = x[:, port_idx_pkts]
                globals()["g%s" % cons_idx] = (
                    np.divide(a, b, out=np.zeros_like(a), where=b != 0) - 1500
                )
                constraints_part.append(globals()["g%s" % cons_idx])
                cons_idx += 1
        return cons_idx, constraints_part
