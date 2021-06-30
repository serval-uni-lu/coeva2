import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_scaler_from_norm(norm, nb_features):

    scaler = MinMaxScaler(feature_range=(0, 1))

    if norm in [2, "2"]:
        scaler.fit([[0], [np.sqrt(nb_features)]])
    elif norm in [np.inf, "inf"]:
        scaler.fit([[0], [1]])
    else:
        raise NotImplementedError

    return scaler
