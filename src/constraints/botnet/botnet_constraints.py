import pickle
from typing import Union

import autograd.numpy as anp
import numpy as np
import tensorflow as tf

from src.attacks.moeva2.constraints.file_constraints import FileConstraints
from src.constraints.augmented_constraints import AugmentedConstraints


class BotnetConstraints(FileConstraints):
    def fix_features_types(self, x) -> Union[np.ndarray, tf.Tensor]:
        return x

    def __init__(self):
        features_path = "./data/botnet/features.csv"
        self.tolerance = 1e-3
        with open("./data/botnet/feat_idx.pickle", "rb") as f:
            self.feat_idx = pickle.load(f)
        self.feat_idx_tf = self.feat_idx.copy()
        for key in self.feat_idx:
            self.feat_idx_tf[key] = tf.convert_to_tensor(
                self.feat_idx[key], dtype=tf.int64
            )
        super().__init__(features_path)

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    def evaluate(self, x: np.ndarray, use_tensors: bool = False) -> np.ndarray:
        if use_tensors:
            return self.evaluate_tf2(x)
        else:
            return self.evaluate_numpy(x)

    def evaluate_tf2(self, x):
        tol = 1e-3

        sum_idx = tf.convert_to_tensor([0, 3, 6, 12, 15, 18], dtype=tf.int64)
        max_idx = tf.convert_to_tensor([1, 4, 7, 13, 16, 19], dtype=tf.int64)
        min_idx = tf.convert_to_tensor([2, 5, 8, 14, 17, 20], dtype=tf.int64)

        g1 = (
            tf.math.abs(
                (
                    tf.math.reduce_sum(
                        tf.gather(x, self.feat_idx_tf["icmp_sum_s_idx"], axis=1), axis=1
                    )
                    + tf.math.reduce_sum(
                        tf.gather(x, self.feat_idx_tf["udp_sum_s_idx"], axis=1), axis=1
                    )
                    + tf.math.reduce_sum(
                        tf.gather(x, self.feat_idx_tf["tcp_sum_s_idx"], axis=1), axis=1
                    )
                )
                - (
                    tf.math.reduce_sum(
                        tf.gather(x, self.feat_idx_tf["bytes_in_sum_s_idx"], axis=1),
                        axis=1,
                    )
                    + tf.math.reduce_sum(
                        tf.gather(x, self.feat_idx_tf["bytes_out_sum_s_idx"], axis=1),
                        axis=1,
                    )
                )
            )
            - 0.4999999
        )
        g2 = tf.math.abs(
            (
                tf.math.reduce_sum(
                    tf.gather(x, self.feat_idx_tf["icmp_sum_d_idx"], axis=1), axis=1
                )
                + tf.math.reduce_sum(
                    tf.gather(x, self.feat_idx_tf["udp_sum_d_idx"], axis=1), axis=1
                )
                + tf.math.reduce_sum(
                    tf.gather(x, self.feat_idx_tf["tcp_sum_d_idx"], axis=1), axis=1
                )
            )
            - (
                tf.math.reduce_sum(
                    tf.gather(x, self.feat_idx_tf["bytes_in_sum_d_idx"], axis=1), axis=1
                )
                + tf.math.reduce_sum(
                    tf.gather(x, self.feat_idx_tf["bytes_out_sum_d_idx"], axis=1),
                    axis=1,
                )
            )
        )

        constraints0 = self.define_individual_constraints_pkts_bytes_tf(
            x, self.feat_idx_tf
        )
        constraints1 = self.define_individual_constraints_tf(
            x, self.feat_idx_tf, sum_idx, max_idx
        )
        constraints2 = self.define_individual_constraints_tf(
            x, self.feat_idx_tf, sum_idx, min_idx
        )
        constraints3 = self.define_individual_constraints_tf(
            x, self.feat_idx_tf, max_idx, min_idx
        )

        constraints = tf.stack(
            [g1, g2] + constraints0 + constraints1 + constraints2 + constraints3, 1
        )

        constraints = tf.clip_by_value(constraints - tol, 0, tf.constant(np.inf))

        return constraints

    def evaluate_numpy(self, x: np.ndarray) -> np.ndarray:
        # ----- PARAMETERS

        tol = 1e-3
        # should write a function in utils for this part

        sum_idx = [0, 3, 6, 12, 15, 18]
        max_idx = [1, 4, 7, 13, 16, 19]
        min_idx = [2, 5, 8, 14, 17, 20]

        g1 = np.absolute(
            (
                x[:, self.feat_idx["icmp_sum_s_idx"]].sum(axis=1)
                + x[:, self.feat_idx["udp_sum_s_idx"]].sum(axis=1)
                + x[:, self.feat_idx["tcp_sum_s_idx"]].sum(axis=1)
            )
            - (
                x[:, self.feat_idx["bytes_in_sum_s_idx"]].sum(axis=1)
                + x[:, self.feat_idx["bytes_out_sum_s_idx"]].sum(axis=1)
            )
        )
        g2 = np.absolute(
            (
                x[:, self.feat_idx["icmp_sum_d_idx"]].sum(axis=1)
                + x[:, self.feat_idx["udp_sum_d_idx"]].sum(axis=1)
                + x[:, self.feat_idx["tcp_sum_d_idx"]].sum(axis=1)
            )
            - (
                x[:, self.feat_idx["bytes_in_sum_d_idx"]].sum(axis=1)
                + x[:, self.feat_idx["bytes_out_sum_d_idx"]].sum(axis=1)
            )
        )

        constraints = [g1, g2]

        cons_idx = 3
        cons_idx, constraints0 = self.define_individual_constraints_pkts_bytes(
            x, cons_idx, self.feat_idx
        )
        constraints.extend(constraints0)
        cons_idx, constraints1 = self.define_individual_constraints(
            x, cons_idx, self.feat_idx, sum_idx, max_idx
        )
        constraints.extend(constraints1)
        cons_idx, constraints2 = self.define_individual_constraints(
            x, cons_idx, self.feat_idx, sum_idx, min_idx
        )
        constraints.extend(constraints2)
        cons_idx, constraints3 = self.define_individual_constraints(
            x, cons_idx, self.feat_idx, max_idx, min_idx
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

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask

    @staticmethod
    def define_individual_constraints_tf(x, feat_idx, upper_idx, lower_idx):
        constraints = []

        keys = list(feat_idx.keys())
        for i in range(len(upper_idx)):
            key = keys[upper_idx[i]]
            type_lower = keys[lower_idx[i]]
            type_upper = keys[upper_idx[i]]
            for j in range(len(feat_idx[key])):
                port_idx_lower = feat_idx[type_lower][j]
                port_idx_upper = feat_idx[type_upper][j]
                constraints.append(x[:, port_idx_lower] - x[:, port_idx_upper])
        return constraints

    @staticmethod
    def define_individual_constraints_pkts_bytes_tf(x, feat_idx):
        constraints = []
        alpha = 1e-5
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
                broken = a - b
                ratio = a / (b + alpha)
                clean_ratio = tf.where(
                    tf.math.is_nan(broken), tf.zeros_like(ratio), ratio
                )
                constraints.append(clean_ratio)
        return constraints

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


class BotnetAugmentedConstraints(AugmentedConstraints):
    def __init__(self):
        important_features = np.load("./data/botnet_augmented/important_features.npy")
        super().__init__(BotnetConstraints(), important_features)


class BotnetRfAugmentedConstraints(AugmentedConstraints):
    def __init__(self):
        important_features = np.load("./data/botnet_rf_augmented/important_features.npy")
        super().__init__(BotnetConstraints(), important_features)
