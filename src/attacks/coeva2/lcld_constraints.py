from typing import Tuple
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from attacks.coeva2.constraints import Constraints
import autograd.numpy as anp
import pandas as pd


class LcldConstraints(Constraints):
    def __init__(
        self,
        amount_feature_index: int,
        constraints_min=None,
        constraints_max=None,
        feature_min=None,
        feature_max=None,
        mutable_mask=None,
        feature_type=None,
    ):
        self._constraints_min = constraints_min
        self._constraints_max = constraints_max
        self._feature_min = feature_min
        self._feature_max = feature_max
        self._mutable_mask = mutable_mask
        self._feature_type = feature_type
        self._scaler = None
        self._amount_feature_index = amount_feature_index

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

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        calculated_installment = (
            np.ceil(
                100
                * (x[:, 0] * (x[:, 2] / 1200) * (1 + x[:, 2] / 1200) ** x[:, 1])
                / ((1 + x[:, 2] / 1200) ** x[:, 1] - 1)
            )
            / 100
        )
        g41 = np.absolute(x[:, 3] - calculated_installment)

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

    def get_feature_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._feature_min, self._feature_max

    def get_feature_type(self) -> np.ndarray:
        return self._feature_type

    def get_amount_feature_index(self) -> int:
        return self._amount_feature_index

    def provision_feature_constraints(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._feature_min = df["min"].to_numpy()
        self._feature_max = df["max"].to_numpy()
        self._mutable_mask = df["mutable"].to_numpy()
        self._feature_type = df["type"].to_numpy()

    def provision_constraints_min_max(self, path: str) -> None:
        df = pd.read_csv(path, low_memory=False)
        self._constraints_min = df["min"].to_numpy()
        self._constraints_max = df["max"].to_numpy()
        self._fit_scaler()
