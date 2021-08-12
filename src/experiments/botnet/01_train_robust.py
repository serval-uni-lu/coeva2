import os
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from src.config_parser.config_parser import get_config
from src.experiments.botnet.features import augment_data
from src.experiments.botnet.model import train_model, print_score
from src.utils.in_out import load_model

np.random.seed(205)
import tensorflow as tf

tf.random.set_seed(206)

from sklearn.preprocessing import MinMaxScaler

# ----- CONFIG
config = get_config()
project_name = "botnet"
nb_important_features = 5
threshold = 0.5

# ----- LOAD

x_train = np.load(f"./data/{project_name}/X_train.npy")
x_test = np.load(f"./data/{project_name}/X_test.npy")
y_train = np.load(f"./data/{project_name}/y_train.npy")
y_test = np.load(f"./data/{project_name}/y_test.npy")
features = pd.read_csv(f"./data/{project_name}/features.csv")
constraints = pd.read_csv(f"./data/{project_name}/constraints.csv")

# ----- SCALER

scaler_path = f"./models/{project_name}/scaler.joblib"
if os.path.exists(scaler_path):
    print(f"{scaler_path} exists loading...")
    scaler = joblib.load(scaler_path)
else:
    scaler = MinMaxScaler()
    x_all = np.concatenate((x_train, x_test))
    x_min = features["min"]
    x_max = features["max"]
    x_min[x_min == "dynamic"] = np.min(x_all, axis=0)[x_min == "dynamic"]
    x_max[x_max == "dynamic"] = np.max(x_all, axis=0)[x_max == "dynamic"]
    x_min = x_min.astype(np.float).to_numpy().reshape(1, -1)
    x_max = x_max.astype(np.float).to_numpy().reshape(1, -1)
    x_min = np.min(np.concatenate((x_min, x_all)), axis=0).reshape(1, -1)
    x_max = np.max(np.concatenate((x_max, x_all)), axis=0).reshape(1, -1)
    scaler.fit(np.concatenate((np.floor(x_min), np.ceil(x_max))))
    joblib.dump(scaler, scaler_path)

# ----- TRAIN MODEL

model_path = f"./models/{project_name}/nn.model"

if os.path.exists(model_path):
    print(f"{model_path} exists loading...")
    model = load_model(model_path)
else:
    model = train_model(scaler.transform(x_train), y_train)
    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )
# ----- MODEL SCORE

y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
y_pred = (y_proba >= threshold).astype(int)
print_score(y_test, y_pred)

# ----- FIND IMPORTANT FEATURES

important_features_path = f"./data/{project_name}/important_features.npy"
if os.path.exists(important_features_path):
    print(f"{important_features_path} exists loading...")
    important_features = np.load(important_features_path)
else:
    sampler = RandomUnderSampler(sampling_strategy={0: 300, 1: 300}, random_state=42)
    x_train_small, y_train_small = sampler.fit_resample(x_train, y_train)
    explainer = shap.DeepExplainer(model, scaler.transform(x_train_small))
    shap_values = explainer.shap_values(scaler.transform(x_train_small))
    shap_values_per_feature = np.mean(np.abs(np.array(shap_values)[0]), axis=0)
    shap_values_per_mutable_feature = shap_values_per_feature[features["mutable"]]

    mutable_feature_index = np.where(features["mutable"])[0]
    order_feature_mutable = np.argsort(shap_values_per_mutable_feature)[::-1]
    important_features_index = mutable_feature_index[order_feature_mutable][
        :nb_important_features
    ]
    important_features_mean = np.mean(x_train[:, important_features_index], axis=0)

    important_features = np.column_stack(
        [important_features_index, important_features_mean]
    )
    np.save(important_features_path, important_features)


# ----- AUGMENT DATASET
x_train_augmented_path = f"./data/{project_name}_augmented/x_train.npy"
x_test_augmented_path = f"./data/{project_name}_augmented/x_test.npy"
features_augmented_path = f"./data/{project_name}_augmented/features.csv"
constraints_augmented_path = f"./data/{project_name}_augmented/constraints.csv"
if os.path.exists(x_train_augmented_path) and os.path.exists(x_test_augmented_path):
    x_train_augmented = np.load(x_train_augmented_path)
    x_test_augmented = np.load(x_test_augmented_path)
    features_augmented = pd.read_csv(features_augmented_path)
    constraints_augmented = pd.read_csv(constraints_augmented_path)
    nb_new_features = x_train_augmented.shape[1] - x_train.shape[1]
else:
    x_train_augmented = augment_data(x_train, important_features)
    x_test_augmented = augment_data(x_test, important_features)
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
    np.save(x_train_augmented_path, x_train_augmented)
    np.save(x_test_augmented_path, x_test_augmented)
    features_augmented.to_csv(features_augmented_path)
    constraints_augmented.to_csv(constraints_augmented_path)

# ----- Augmented scaler

scaler_augmented_path = f"./models/{project_name}_augmented/scaler.joblib"

if os.path.exists(scaler_augmented_path):
    scaler_augmented = joblib.load(scaler_augmented_path)
else:
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

# ----- TRAIN MODEL

model_augmented_path = f"./models/{project_name}_augmented/nn.model"

if os.path.exists(model_augmented_path):
    print(f"{model_augmented_path} exists loading...")
    model_augmented = load_model(model_augmented_path)
else:
    model_augmented = train_model(
        scaler_augmented.transform(x_train_augmented), y_train
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
# ----- MODEL SCORE

y_proba = model_augmented.predict_proba(
    scaler_augmented.transform(x_test_augmented)
).reshape(-1)
y_pred_augmented = (y_proba >= threshold).astype(int)
print_score(y_test, y_pred)


# ----- Common x_attacks
x_candidates_path = f"./data/{project_name}/x_candidates_common.npy"
x_candidates_augmented_path = f"./data/{project_name}_augmented/x_candidates_common.npy"

if os.path.exists(x_candidates_path) and os.path.exists(x_candidates_augmented_path):
    x_candidates = np.load(x_candidates_path)
    x_candidates_augmented = np.load(x_candidates_augmented_path)
else:
    candidates_index = (y_test == 1) * (y_test == y_pred) * (y_test == y_pred_augmented)
    x_candidates = x_test[candidates_index, :]
    x_candidates_augmented = x_test_augmented[candidates_index, :]
    np.save(x_candidates_path, x_candidates)
    np.save(x_candidates_augmented_path, x_candidates_augmented)
