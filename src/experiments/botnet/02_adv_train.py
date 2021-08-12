import os
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.config_parser.config_parser import get_config
from src.experiments.botnet.features import augment_data
from src.experiments.botnet.model import train_model, print_score
from src.experiments.united.utils import get_constraints_from_str
from src.utils import filter_initial_states
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
moeva_path = ""
gradient_path = ""
norm = 2
candidates_used = [0, 100]

# ----- LOAD

x_train = np.load(f"./data/{project_name}/X_train.npy")
x_test = np.load(f"./data/{project_name}/X_test.npy")
y_train = np.load(f"./data/{project_name}/y_train.npy")
y_test = np.load(f"./data/{project_name}/y_test.npy")
features = pd.read_csv(f"./data/{project_name}/features.csv")
constraints = pd.read_csv(f"./data/{project_name}/constraints.csv")


# ----- LOAD SCALER

scaler_path = f"./models/{project_name}/scaler.joblib"
scaler = joblib.load(scaler_path)

# ----- LOAD MODEL
model_path = f"./models/{project_name}/nn.model"
model = load_model(model_path)

# ----- MODEL SCORE

y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
y_pred = (y_proba >= threshold).astype(int)
print_score(y_test, y_pred)

# ----- LOAD CANDIDATES
x_candidates_path = f"./data/{project_name}/x_candidates_common.npy"
x_candidates = np.load(x_candidates_path)

# ----- LOAD Adversarials
x_moeva = np.load(moeva_path)
gradient = np.load(gradient_path)

# ----- CREATE HELPER OBJECT
classifier = Classifier(model)
constraints_calculator = get_constraints_from_str(project_name)(
    f"./data/{project_name}/features.csv",
    f"./data/{project_name}/constraints.csv",
)
objective_calc = ObjectiveCalculator(
    classifier,
    constraints_calculator,
    minimize_class=1,
    thresholds=config["thresholds"],
    min_max_scaler=scaler,
    ml_scaler=scaler,
    norm=norm,
)

# ----- RETRIEVE ADVS
x_candidates_used = filter_initial_states(
    x_candidates, candidates_used[0], candidates_used[1]
)


def create_adv_model(x_attacks, attack_name):
    x_train_adv_path = f"./data/{project_name}/x_train_{attack_name}.npy"
    y_train_adv_path = f"./data/{project_name}/y_train_{attack_name}.npy"

    if os.path.exists(x_train_adv_path) and os.path.exists(y_train_adv_path):
        x_train_adv = np.load(x_train_adv_path)
        y_train_adv = np.load(y_train_adv_path)

    # ---- RETRIEVE ADV
    x_attacks_adv = objective_calc.get_successful_attacks(
        x_candidates_used, x_moeva, preferred_metrics="distance", order="asc", max_inputs=-1
    )
    # ---- AUGMENT DATA
    x_train_adv = np.concatenate((x_train, x_attacks_adv), axis=0)
    y_train_adv = np.concatenate((y_train, np.zeros(x_attacks_adv.shape[0])), axis=0)



# ---- MOEVA
x_moeva_adv = objective_calc.get_successful_attacks(
    x_candidates_used, x_moeva, preferred_metrics="distance", order="asc", max_inputs=-1
)
x_gradient_adv = objective_calc.get_successful_attacks(
    x_candidates_used, x_gradient_adv, preferred_metrics="distance", order="asc", max_inputs=-1
)

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
