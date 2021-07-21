from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.constraints import Constraints
import autograd.numpy as anp
import pandas as pd
import logging


class LcldConstraints(Constraints):
    def __init__(
        self,
        feature_path: str,
        constraints_path: str,
    ):
        self._provision_constraints_min_max(constraints_path)
        self._provision_feature_constraints(feature_path)
        self._fit_scaler()

    def _fit_scaler(self) -> None:
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        min_c, max_c = self.get_constraints_min_max()
        self._scaler = self._scaler.fit([min_c, max_c])

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    def evaluate_tf2(self, x):
        # ----- PARAMETERS

        import tensorflow as tf
        tol = 1e-3
        alpha = 1e-5

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2] / 1200
        x3  = x[:, 3]

        calculated_installment = (
                (x[:, 0] * (x[:, 2] / 1200) * (1 + x[:, 2] / 1200) ** x[:, 1])
                / ((1 + x[:, 2] / 1200) ** x[:, 1] - 1 + alpha)

        )

        g41 = tf.math.abs(x[:, 3] - calculated_installment) - 0.099999


        calculated_installment_36 = (
                (x0 * x2 * (1 + x2) ** 36)
                    / ((1 + x2) ** 36 - 1)

        )

        calculated_installment_60 = (
                (x0 * x2 * (1 + x2) ** 60)
                / ((1 + x2) ** 36 - 1)

        )

        term_constraint = tf.minimum(tf.math.abs(x3-36)/36,tf.math.abs(x3-60)/60)


        g41_36 = tf.math.abs(x3 - calculated_installment_36)
        g41_60 = tf.math.abs(x3 - calculated_installment_60)

        g41_ = tf.minimum(g41_36*tf.math.abs(x3-36)/36,g41_60*tf.math.abs(x3-60)/60)# - 0.099999

        # open_acc <= total_acc
        g42 = alpha + x[:, 10] - x[:, 14]
        g42 = tf.clip_by_value(g42,0,tf.constant(np.inf))

        # pub_rec_bankruptcies <= pub_rec
        g43 = tf.clip_by_value(alpha + x[:, 16] - x[:, 11],0,tf.constant(np.inf))

        # term = 36 or term = 60
        g44 = tf.math.minimum(tf.math.abs(36 - x[:, 1]), tf.math.abs(60 - x[:, 1]))

        # ratio_loan_amnt_annual_inc
        g45 = tf.math.abs(x[:, 20] - x[:, 0] / x[:, 6])

        # ratio_open_acc_total_acc
        g46 = tf.math.abs(x[:, 21] - x[:, 10] / x[:, 14])

        # diff_issue_d_earliest_cr_line
        g47 = tf.math.abs(
            x[:, 22]
            - (
                    self._date_feature_to_month(x[:, 7])
                    - self._date_feature_to_month(x[:, 9])
            )
        )

        # ratio_pub_rec_diff_issue_d_earliest_cr_line
        g48 = tf.math.abs(x[:, 23] - x[:, 11] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        g49 = tf.math.abs(x[:, 24] - x[:, 16] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        # ratio_mask = x[:, 11] == 0
        # ratio = torch.empty(x.shape[0])
        # ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
        # ratio[~ratio_mask] = x[~ratio_mask, 16] / x[~ratio_mask, 11]
        # ratio[ratio == np.inf] = -1
        # ratio[np.isnan(ratio)] = -1

        broken = x[:, 16] / x[:, 11]
        ratio = x[:, 16] / (x[:, 11]+ alpha)
        #g410 = tf.math.abs(x[:, 25] - x[:, 16] / (x[:, 11] + alpha))
        clean_ratio = tf.where(tf.math.is_nan(broken), -1 * tf.ones_like(ratio), ratio)
        g410 = tf.math.abs(x[:, 25] - clean_ratio)

        constraints = tf.stack([g41,g42,g43,g44,g45,g46,g47,g48,g49,g410],1)

        constraints = tf.clip_by_value(constraints - tol, 0, tf.constant(np.inf))
        return constraints
        # print(max_constraints.cpu().detach())
        #return max_constraints.mean()

    def evaluate(self, x: np.ndarray, use_tensors:bool=False) -> np.ndarray:
        if use_tensors:
            return self.evaluate_tf2(x)
        else:
            return self.evaluate_numpy(x)

    def evaluate_numpy(self, x: np.ndarray) -> np.ndarray:
        # ----- PARAMETERS

        tol = 1e-3

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        calculated_installment = (
            x[:, 0] * (x[:, 2] / 1200) * (1 + x[:, 2] / 1200) ** x[:, 1]
        ) / ((1 + x[:, 2] / 1200) ** x[:, 1] - 1)
        g41 = np.absolute(x[:, 3] - calculated_installment) - 0.099999

        # open_acc <= total_acc
        g42 = x[:, 10] - x[:, 14]

        # pub_rec_bankruptcies <= pub_rec
        g43 = x[:, 16] - x[:, 11]

        # term = 36 or term = 60
        g44 = np.absolute((36 - x[:, 1]) * (60 - x[:, 1]))

        # ratio_loan_amnt_annual_inc
        g45 = np.absolute(x[:, 20] - x[:, 0] / x[:, 6])

        # ratio_open_acc_total_acc
        g46 = np.absolute(x[:, 21] - x[:, 10] / x[:, 14])

        # diff_issue_d_earliest_cr_line
        g47 = np.absolute(
            x[:, 22]
            - (
                self._date_feature_to_month(x[:, 7])
                - self._date_feature_to_month(x[:, 9])
            )
        )

        # ratio_pub_rec_diff_issue_d_earliest_cr_line
        g48 = np.absolute(x[:, 23] - x[:, 11] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        g49 = np.absolute(x[:, 24] - x[:, 16] / x[:, 22])

        # ratio_pub_rec_bankruptcies_pub_rec
        ratio_mask = x[:, 11] == 0
        ratio = np.empty(x.shape[0])
        ratio = np.ma.masked_array(ratio, mask=ratio_mask, fill_value=-1).filled()
        ratio[~ratio_mask] = x[~ratio_mask, 16] / x[~ratio_mask, 11]
        ratio[ratio == np.inf] = -1
        ratio[np.isnan(ratio)] = -1
        g410 = np.absolute(x[:, 25] - ratio)

        constraints = anp.column_stack(
            [g41, g42, g43, g44, g45, g46, g47, g48, g49, g410]
        )
        constraints[constraints <= tol] = 0.0

        return constraints

    def get_nb_constraints(self) -> int:
        return 10

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
