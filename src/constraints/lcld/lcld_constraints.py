import autograd.numpy as anp
import numpy as np
import tensorflow as tf

from src.attacks.moeva2.constraints.file_constraints import FileConstraints
from src.attacks.moeva2.utils import get_ohe_masks
from src.constraints.augmented_constraints import AugmentedConstraints


class LcldConstraints(FileConstraints):
    def __init__(self):
        features_path = "./data/lcld/features.csv"
        self.important_features = np.load("./data/lcld/important_features.npy")

        super().__init__(features_path)

    @staticmethod
    def _date_feature_to_month(feature):
        return np.floor(feature / 100) * 12 + (feature % 100)

    @staticmethod
    def _date_feature_to_month_tf(feature):
        return tf.math.floor(feature / 100) * 12 + tf.math.floormod(feature, 100)

    def fix_features_types(self, x):

        new_tensor_v = tf.Variable(x)

        # enforcing 2 possibles values
        x1 = tf.where(x[:, 1] < (60 + 36) / 2, 36 * tf.ones_like(x[:, 1]), 60)
        new_tensor_v = new_tensor_v[:, 1].assign(x1)

        x0 = x[:, 0]
        x2 = x[:, 2] / 1200

        # enforcing the power formula
        x3 = x0 * x2 * tf.math.pow(1 + x2, x1) / (tf.math.pow(1 + x2, x1) - 1)
        new_tensor_v = new_tensor_v[:, 3].assign(x3)

        # enforcing ohe

        ohe_masks = get_ohe_masks(self._feature_type)

        new_tensor_v = new_tensor_v.numpy()
        for mask in ohe_masks:
            ohe = new_tensor_v[:, mask]
            max_feature = np.argmax(ohe, axis=1)
            new_ohe = np.zeros_like(ohe)
            new_ohe[np.arange(len(ohe)), max_feature] = 1

            new_tensor_v[:, mask] = new_ohe

        return tf.convert_to_tensor(new_tensor_v, dtype=tf.float32)

    def evaluate_tf2(self, x):
        # ----- PARAMETERS

        tol = 1e-3
        alpha = 1e-5

        # installment = loan_amount * int_rate (1 + int_rate) ^ term / ((1+int_rate) ^ term - 1)
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2] / 1200
        x3 = x[:, 3]

        calculated_installment = (
            x0 * x2 * tf.math.pow(1 + x2, x1) / (tf.math.pow(1 + x2, x1) - 1)
        )

        calculated_installment_36 = (
            x0 * x2 * tf.math.pow(1 + x2, 36) / (tf.math.pow(1 + x2, 36) - 1)
        )

        calculated_installment_60 = (
            x0 * x2 * tf.math.pow(1 + x2, 60) / (tf.math.pow(1 + x2, 60) - 1)
        )

        g41 = (
            tf.minimum(
                tf.math.abs(x3 - calculated_installment_36),
                tf.math.abs(x3 - calculated_installment_60),
            )
            - 0.099999
        )
        g41_ = tf.math.abs(x3 - calculated_installment) - 0.099999

        # open_acc <= total_acc
        g42 = alpha + x[:, 10] - x[:, 14]
        g42 = tf.clip_by_value(g42, 0, tf.constant(np.inf))

        # pub_rec_bankruptcies <= pub_rec
        g43 = tf.clip_by_value(alpha + x[:, 16] - x[:, 11], 0, tf.constant(np.inf))

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
                self._date_feature_to_month_tf(x[:, 7])
                - self._date_feature_to_month_tf(x[:, 9])
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
        ratio = x[:, 16] / (x[:, 11] + alpha)
        # g410 = tf.math.abs(x[:, 25] - x[:, 16] / (x[:, 11] + alpha))
        clean_ratio = tf.where(tf.math.is_nan(broken), -1 * tf.ones_like(ratio), ratio)
        g410 = tf.math.abs(x[:, 25] - clean_ratio)

        constraints = tf.stack([g41, g42, g43, g44, g45, g46, g47, g48, g49, g410], 1)

        constraints = tf.clip_by_value(constraints - tol, 0, tf.constant(np.inf))

        return constraints
        # return tf.nn.softmax(constraints) * tf.reduce_max(constraints)
        # print(max_constraints.cpu().detach())
        # return max_constraints.mean()

    def evaluate(self, x: np.ndarray, use_tensors: bool = False) -> np.ndarray:
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


class LcldAugmentedConstraints(AugmentedConstraints):
    def __init__(self):
        important_features = np.load("./data/lcld_augmented/important_features.npy")
        super().__init__(LcldConstraints(), important_features)


class LcldRfAugmentedConstraints(AugmentedConstraints):
    def __init__(self):
        important_features = np.load("./data/lcld_rf_augmented/important_features.npy")
        super().__init__(LcldConstraints(), important_features)
