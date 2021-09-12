from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.config_parser.config_parser import get_config
from src.experiments.botnet.features import augment_data

from src.utils.in_out import load_model

config = get_config()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Parameters

project_name = config.get("project_name")
threshold = config["thresholds"]["misclassification"]
if project_name == "lcld":
    from src.experiments.lcld.model import print_score, train_model
elif project_name == "botnet":
    from src.experiments.botnet.model import print_score, train_model

# Load data

x_train = np.load(f"./data/{project_name}/X_train.npy")
x_test = np.load(f"./data/{project_name}/X_test.npy")
y_train = np.load(f"./data/{project_name}/y_train.npy")
y_test = np.load(f"./data/{project_name}/y_test.npy")
features = pd.read_csv(f"./data/{project_name}/features.csv")
constraints = pd.read_csv(f"./data/{project_name}/constraints.csv")
x_candidates = np.load(f"./data/{project_name}/x_candidates_common.npy")
# Load scaler and models

scaler_path = f"./models/{project_name}/scaler.joblib"
print(f"{scaler_path} exists loading...")
scaler = joblib.load(scaler_path)
model_path = f"./models/{project_name}/nn.model"
model = load_model(model_path)

y_proba = model.predict_proba(scaler.transform(x_test))
y_pred = (y_proba[:, 1] >= threshold).astype(int)
print_score(y_test, y_pred)
print(f"AUROC: {roc_auc_score(y_test, y_proba[:, 1])}")

correctly_classified_1 = (
    model.predict_proba(scaler.transform(x_candidates))[:, 1] >= threshold
).astype(int)

# Find important features
important_features_index_path = (
    f"./data/{project_name}/sm_constraints/important_features_index.npy"
)
if os.path.exists(important_features_index_path):
    print(f"{important_features_index_path} exists loading...")
    important_features_index = np.load(important_features_index_path)
else:
    sampler = RandomUnderSampler(sampling_strategy={0: 2500, 1: 2500}, random_state=42)
    x_train_small, y_train_small = sampler.fit_resample(x_train, y_train)
    explainer = shap.DeepExplainer(model, scaler.transform(x_train_small))
    shap_values = explainer.shap_values(scaler.transform(x_train_small))
    shap_values_per_feature = np.mean(np.abs(np.array(shap_values)[0]), axis=0)
    shap_values_per_mutable_feature = shap_values_per_feature[features["mutable"]]

    mutable_feature_index = np.where(features["mutable"])[0]
    order_feature_mutable = np.argsort(shap_values_per_mutable_feature)[::-1]

    important_features_index = mutable_feature_index[order_feature_mutable]
    np.save(important_features_index_path, important_features_index)

# For i in important features, train model, and save correctly classified
for nb_features in np.arange(2, min(27, len(important_features_index))):
    important_features_path = (
        f"./data/{project_name}/sm_constraints/important_features_{nb_features}.npy"
    )
    important_features_index_local = important_features_index[:nb_features]
    important_features_mean = np.mean(
        x_train[:, important_features_index_local], axis=0
    )

    important_features = np.column_stack(
        [important_features_index_local, important_features_mean]
    )
    np.save(important_features_path, important_features)

    x_train_augmented = augment_data(x_train, important_features)
    x_test_augmented = augment_data(x_test, important_features)
    x_candidates_augmented = augment_data(x_candidates, important_features)
    nb_new_features = x_train_augmented.shape[1] - x_train.shape[1]
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
    constraints_augmented = constraints.append(
        [
            {
                "min": 0.0,
                "max": 1.0,
                "augmentation": True,
            }
            for i in range(nb_new_features)
        ]
    )
    features_augmented_path = f"./data/{project_name}/sm_constraints/features_augmented_{nb_features}.csv"
    constraints_augmented_path = f"./data/{project_name}/sm_constraints/constraints_augmented_{nb_features}.csv"
    features_augmented.to_csv(features_augmented_path)
    constraints_augmented.to_csv(constraints_augmented_path)
    scaler_augmented_path = (
        f"./models/{project_name}/sm_constraints/scaler_augmented_{nb_features}.joblib"
    )
    scaler_augmented = MinMaxScaler()
    scaling_data = np.concatenate(
        (
            np.concatenate((scaler.data_min_, np.zeros(nb_new_features))),
            np.concatenate((scaler.data_max_, np.ones(nb_new_features))),
        ),
        axis=0,
    ).reshape(2, -1)

    scaler_augmented.fit(scaling_data)
    joblib.dump(scaler_augmented, scaler_augmented_path)

    model_augmented_path = (
        f"./models/{project_name}/sm_constraints/nn_augmented_{nb_features}.model"
    )
    if os.path.exists(model_augmented_path):
        print(f"{model_augmented_path} exists loading...")
        model_augmented = load_model(model_augmented_path)
    else:
        model_augmented = train_model(
            scaler_augmented.transform(x_train_augmented), to_categorical(y_train)
        )
        tf.keras.models.save_model(
            model_augmented,
            model_augmented_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
        )
        y_proba = model_augmented.predict_proba(
            scaler_augmented.transform(x_test_augmented)
        )
        y_pred_augmented = (y_proba[:, 1] >= threshold).astype(int)
        print_score(y_test, y_pred_augmented)
        print(f"AUROC: {roc_auc_score(y_test, y_proba[:, 1])}")

    correctly_classified_1 = correctly_classified_1 * (
        model_augmented.predict_proba(
            scaler_augmented.transform(x_candidates_augmented)
        )[:, 1]
        >= threshold
    ).astype(int)

x_candidates = x_candidates[correctly_classified_1]
print(x_candidates.shape)

for nb_features in np.arange(2, min(27, len(important_features_index))):
    important_features_path = (
        f"./data/{project_name}/sm_constraints/important_features_{nb_features}.npy"
    )
    important_features = np.load(important_features_path)
    x_candidates_augmented = augment_data(x_candidates, important_features)
    np.save(f"./data/{project_name}/sm_constraints/x_candidates_augmented_{nb_features}.npy", x_candidates_augmented)
