from itertools import combinations

import numpy as np
import shap
from imblearn.under_sampling import RandomUnderSampler


def calc_n_important_features(n_features, ratio):
    a, b, c = 0.5, -0.5, -n_features * ratio
    n_important_features = np.floor(np.roots([a, b, c])[0])
    return int(n_important_features)


def find_important_features(model, scaler, mutable_mask, X, y, n_important_features):

    sampler = RandomUnderSampler(sampling_strategy={0: 500, 1: 500}, random_state=42)
    x_small, y_small = sampler.fit_resample(X, y)

    explainer = shap.DeepExplainer(model, scaler.transform(x_small))
    shap_values = explainer.shap_values(scaler.transform(x_small))
    shap_values_per_feature = np.mean(np.abs(np.array(shap_values)[0]), axis=0)
    shap_values_per_mutable_feature = shap_values_per_feature[mutable_mask]

    mutable_feature_index = np.where(mutable_mask)[0]
    order_feature_mutable = np.argsort(shap_values_per_mutable_feature)[::-1]
    important_features_index = mutable_feature_index[order_feature_mutable][
        :n_important_features
    ]

    important_features_mean = np.mean(X[:, important_features_index], axis=0)
    important_features = np.column_stack(
        [important_features_index, important_features_mean]
    )
    return important_features


def augment_data(x, important_features):
    original_shape = x.shape
    local_x = x.reshape(-1, original_shape[-1])

    comb = combinations(range(important_features.shape[0]), 2)
    new_features = []
    for i1, i2 in comb:
        new_features.append(
            np.logical_xor(
                (
                    local_x[:, int(important_features[i1, 0])]
                    >= important_features[i1, 1]
                ),
                (
                    local_x[:, int(important_features[i2, 0])]
                    >= important_features[i2, 1]
                ),
            ).astype(np.float64)
        )
    new_features = np.column_stack(new_features)
    new_x = np.concatenate((local_x, new_features), axis=1).reshape(
        *original_shape[:-1], -1
    )
    return new_x


def augment_dataset(model, scaler, features, X_train, y_train, X_test, ratio):

    n_features = X_train.shape[1]
    n_important_features = calc_n_important_features(n_features, ratio)

    mutable_mask = features["mutable"]
    important_features = find_important_features(
        model, scaler, mutable_mask, X_train, y_train, n_important_features
    )
    X_train_augmented = augment_data(X_train, important_features)
    X_test_augmented = augment_data(X_test, important_features)
    nb_new_features = X_train_augmented.shape[1] - X_train.shape[1]
    features_augmented = features.append(
        [
            {
                "feature": f"augmented_{i}",
                "type": "int",
                "mutable": True,
                "min": 0.0,
                "max": 1.0,
                "augmentation": True,
            }
            for i in range(nb_new_features)
        ]
    )
    return X_train_augmented, X_test_augmented, features_augmented, important_features
