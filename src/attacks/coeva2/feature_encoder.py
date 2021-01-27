from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ONEHOT_ENCODE_KEY = "ohe"


class FeatureEncoder:
    def __init__(
        self,
        mutable_mask: np.ndarray,
        type_mask: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> None:
        self.type_mask = type_mask
        self.mutable_mask = mutable_mask
        self._xl = xl
        self._xu = xu
        self._min_max_scaler = MinMaxScaler()
        self._min_max_scaler.fit(np.array([xl, xu]))

        self._create_one_hot_encoders()

        x_ml_shape1 = mutable_mask.shape[0]
        if (
            type_mask.shape[0] != x_ml_shape1
            or xl.shape[0] != x_ml_shape1
            or xu.shape[0] != x_ml_shape1
        ):
            raise ValueError(
                f"{mutable_mask.__name__}  ({mutable_mask.shape}),"
                f"{type_mask.__name__}     ({type_mask.shape}),"
                f"{xl.__name__}            ({xl.shape}) and "
                f"{xu.__name__}            ({xu.shape})"
                f"must have same shape."
            )

    def _create_one_hot_encoders(self):

        seen_key = []
        one_hot_masks = []

        mutable_type_mask = self._ml_to_mutable(self.type_mask)

        for i, e_type in enumerate(mutable_type_mask):
            if e_type.startswith(ONEHOT_ENCODE_KEY):
                if e_type in seen_key:
                    index = seen_key.index(e_type)
                    one_hot_masks[index].append(i)
                else:
                    seen_key.append(e_type)
                    one_hot_masks.append([i])

        one_hot_masks = [np.array(e) for e in one_hot_masks]

        encoders = [OneHotEncoder(sparse=False) for _ in one_hot_masks]
        for index, one_hot_mask in enumerate(one_hot_masks):
            encoders[index].fit(np.arange(one_hot_mask.shape[0]).reshape(-1, 1))

        self._encoders = encoders
        self._one_hot_masks = one_hot_masks

        no_change_mask = np.ones(mutable_type_mask.shape[0], np.bool)
        for mask in self._one_hot_masks:
            no_change_mask[mask] = 0
        self._no_one_hot_mask = no_change_mask

    def _ml_to_mutable(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.mutable_mask]

    def _mutable_to_ml(self, x: np.ndarray, x_initial_ml) -> np.ndarray:
        x_return = np.zeros((x.shape[0], x_initial_ml.shape[0]))
        x_return[:, ~self.mutable_mask] = x_initial_ml[~self.mutable_mask]
        x_return[:, self.mutable_mask] = x
        return x_return

    def _mutable_to_cat_encode(self, x: np.ndarray) -> np.ndarray:

        result = np.array(x.shape[0], self.get_genetic_v_length())

        # Put the not one encoded feature at the beginning
        result[:, : self._no_one_hot_mask.sum()] = x[:, self._no_one_hot_mask]

        # Put the cat value at the end
        for index, mask in self._one_hot_masks:
            result[:, self._no_one_hot_mask.sum() + index] = self._encoders[
                index
            ].inverse_transform(x[:, mask])

        return result

    def _cat_encode_to_mutable(self, x: np.ndarray) -> np.ndarray:

        result = np.array(x.shape[0], self.mutable_mask.shape[0])
        # Put back the first element back in their place
        result[:, self._no_one_hot_mask] = x[:, : self._no_one_hot_mask.sum()]

        # Put back the scale values element back in their place
        for index, mask in self._one_hot_masks:
            result[:, mask] = self._encoders[index].transform(
                x[:, self._no_one_hot_mask.sum() + index]
            )

        return result

    def ml_to_genetic(self, x: np.ndarray) -> np.ndarray:
        return self._mutable_to_cat_encode(self._ml_to_mutable(x))

    def genetic_to_ml(self, x: np.ndarray, x_initial_ml) -> np.ndarray:
        return self._mutable_to_ml(self._cat_encode_to_mutable(x), x_initial_ml)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return self._min_max_scaler.transform(x)

    def get_min_max_genetic(self) -> Tuple[np.ndarray, np.ndarray]:

        min_max = np.array(self._ml_to_mutable(self._xl), self._ml_to_mutable(self._xu))
        result = np.array(min_max.shape[0], self.get_genetic_v_length())

        # Put the not one encoded feature at the beginning
        result[:, : self._no_one_hot_mask.sum()] = min_max[:, self._no_one_hot_mask]

        min_one_hot = np.zeros(len(self._one_hot_masks))
        max_one_hot = np.array([mask.shape[0] for mask in self._one_hot_masks])

        for index, mask in self._one_hot_masks:
            result[:, self._no_one_hot_mask.sum() + index] = np.array(
                [[min_one_hot[index]], [max_one_hot[index]]]
            )

        return result[0], result[1]

    def get_genetic_v_length(self) -> int:
        encoder_sum = np.array([len(mask) for mask in self._one_hot_masks]).sum()
        return self._no_one_hot_mask.sum() + encoder_sum

    def get_type_mask_genetic(self) -> np.ndarray:

        result = np.array(self.type_mask.shape[0], self.get_genetic_v_length())

        # Put the not one encoded feature at the beginning
        result[:, : self._no_one_hot_mask.sum()] = self.type_mask[
            :, self._no_one_hot_mask
        ]

        for index, mask in self._one_hot_masks:
            result[:, self._no_one_hot_mask.sum() + index] = np.array("int")

        return result
