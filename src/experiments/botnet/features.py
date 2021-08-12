from itertools import combinations

import numpy as np


def augment_data(x, important_features):
    comb = combinations(range(important_features.shape[0]), 2)
    new_features = []
    for i1, i2 in comb:
        new_features.append(
            np.logical_xor(
                (x[:, int(important_features[i1, 0])] >= important_features[i1, 1]),
                (x[:, int(important_features[i2, 0])] >= important_features[i2, 1]),
            ).astype(np.float64)
        )
    new_features = np.column_stack(new_features)
    return np.concatenate((x, new_features), axis=1)
