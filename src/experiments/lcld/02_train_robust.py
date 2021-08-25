import os
import comet_ml
import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints
from src.attacks.moeva2.moeva2 import Moeva2
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.attacks.moeva2.utils import results_to_numpy_results
from src.config_parser.config_parser import get_config
from src.experiments.botnet.features import augment_data
from src.experiments.lcld.model import train_model, print_score
from src.experiments.united.utils import get_constraints_from_str
from src.utils.comet import init_comet
from src.utils.in_out import load_model
from src.attacks.pgd.classifier import TF2Classifier as kc
from src.attacks.pgd.atk import PGDTF2 as PGD

np.random.seed(205)
import tensorflow as tf

tf.random.set_seed(206)

from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----- CONFIG
config = get_config()
project_name = config["project_name"]
nb_important_features = 5
threshold = config["misclassification_threshold"]

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
    raise FileNotFoundError

# ----- TRAIN MODEL

model_path = f"./models/{project_name}/nn.model"

if os.path.exists(model_path):
    print(f"{model_path} exists loading...")
    model = load_model(model_path)
else:
    raise FileNotFoundError
# ----- MODEL SCORE

# y_proba = model.predict_proba(scaler.transform(x_test))
# y_pred = (y_proba[:, 1] >= threshold).astype(int)
# print_score(y_test, y_pred)
# print_score(y_train, (model.predict_proba(scaler.transform(x_train))[:, 1] >= threshold).astype(int))

# ----- FIND IMPORTANT FEATURES

important_features_path = f"./data/{project_name}/important_features.npy"
if os.path.exists(important_features_path):
    print(f"{important_features_path} exists loading...")
    important_features = np.load(important_features_path)
else:
    raise FileNotFoundError


# ----- AUGMENT DATASET
x_train_augmented_path = f"./data/{project_name}/x_train_augmented.npy"
x_test_augmented_path = f"./data/{project_name}/x_test_augmented.npy"
features_augmented_path = f"./data/{project_name}/features_augmented.csv"
constraints_augmented_path = f"./data/{project_name}/constraints_augmented.csv"
if os.path.exists(x_train_augmented_path) and os.path.exists(x_test_augmented_path):
    x_train_augmented = np.load(x_train_augmented_path)
    x_test_augmented = np.load(x_test_augmented_path)
    features_augmented = pd.read_csv(features_augmented_path)
    constraints_augmented = pd.read_csv(constraints_augmented_path)
    nb_new_features = x_train_augmented.shape[1] - x_train.shape[1]
else:
    raise FileNotFoundError

# ----- Augmented scaler

scaler_augmented_path = f"./models/{project_name}/scaler_augmented.joblib"

if os.path.exists(scaler_augmented_path):
    scaler_augmented = joblib.load(scaler_augmented_path)
else:
    raise FileNotFoundError

# ----- TRAIN MODEL

model_augmented_path = f"./models/{project_name}/nn_augmented.model"

if os.path.exists(model_augmented_path):
    print(f"{model_augmented_path} exists loading...")
    model_augmented = load_model(model_augmented_path)
else:
    raise FileNotFoundError

# ----- MODEL SCORE

# y_proba = model_augmented.predict_proba(scaler_augmented.transform(x_test_augmented))
# y_pred_augmented = (y_proba[:, 1] >= threshold).astype(int)
# print_score(y_test, y_pred_augmented)

# ----- ADVERSARIAL CANDIDATES
x_train_candidates = x_train[
        (y_train == 1)
        * (
                y_train
                == (
                        model.predict_proba(scaler.transform(x_train))[:, 1] >= threshold
                ).astype(int)
        )
        ]
constraints = get_constraints_from_str(config["project_name"])(
    config["paths"]["features"],
    config["paths"]["constraints"],
)

constraints_satisfied = np.max(constraints.evaluate(x_train_candidates), axis=1) <= 0
x_train_candidates = x_train_candidates[constraints_satisfied]
print(x_train_candidates.shape)


# ----- ADVERSARIAL GENERATION MOEVA

x_train_moeva_path = f"./data/{project_name}/x_train_moeva.npy"

if os.path.exists(x_train_moeva_path):
    x_train_moeva = np.load(x_train_moeva_path)
else:
    raise FileNotFoundError


# ----- ADVERSARIAL SUCCESS MOEVA
x_train_adv_moeva_path = f"./data/{project_name}/x_train_adv_moeva.npy"

if os.path.exists(x_train_adv_moeva_path):
    x_train_adv_moeva = np.load(x_train_adv_moeva_path)
    x_train_adv_moeva_index = np.load(f"./data/{project_name}/x_train_adv_moeva_index.npy")
else:
    raise FileNotFoundError

#
# # ----- ADVERSARIAL TRAINING MOEVA
#
# model_adv_moeva_path = f"./models/{project_name}/nn_moeva_full.model"
# if os.path.exists(model_adv_moeva_path):
#     model_adv_moeva = load_model(model_adv_moeva_path)
# else:
#     x_train_local = np.concatenate((x_train, x_train_adv_moeva), axis=0)
#     y_train_local = np.concatenate(
#         (y_train, np.ones(x_train_adv_moeva.shape[0])), axis=0
#     )
#
#     model_adv_moeva = train_model(
#         scaler.transform(x_train_local), to_categorical(y_train_local)
#     )
#     tf.keras.models.save_model(
#         model_adv_moeva,
#         model_adv_moeva_path,
#         overwrite=True,
#         include_optimizer=True,
#         save_format=None,
#         signatures=None,
#         options=None,
#     )
#
# y_proba = model_adv_moeva.predict_proba(scaler.transform(x_test))
# y_pred_adv_moeva = (y_proba[:, 1] >= threshold).astype(int)
# print_score(y_test, y_pred_adv_moeva)




constraints = get_constraints_from_str(f"{config['project_name']}_augmented")(
    config["paths"]["features_augmented"],
    config["paths"]["constraints_augmented"],
)

x_train_augmented_candidates = x_train_augmented[
        (y_train == 1)
        * (
                y_train
                == (
                        model_augmented.predict_proba(scaler_augmented.transform(x_train_augmented))[:, 1] >= threshold
                ).astype(int)
        )
    ]
constraints_satisfied = np.max(constraints.evaluate(x_train_augmented_candidates), axis=1) <= 0
x_train_augmented_candidates = x_train_augmented_candidates[constraints_satisfied]
print(x_train_augmented_candidates.shape)


# ----- ADVERSARIAL GENERATION MOEVA AUGMENTED

batch_i = config.get("iter_i")
input_per_batch = 2000

x_train_augmented_moeva_path = f"./data/{project_name}/x_train_augmented_moeva_{batch_i}.npy"

if os.path.exists(x_train_augmented_moeva_path):
    x_train_augmented_moeva = np.load(x_train_augmented_moeva_path)
else:
    print(f"{x_train_augmented_candidates.shape} candidates.")
    n_gen = config["budget"]

    x_train_augmented_candidates = x_train_augmented_candidates[batch_i*input_per_batch: (batch_i+1)*input_per_batch]
    moeva = Moeva2(
        model_augmented_path,
        constraints,
        problem_class=None,
        l2_ball_size=0.0,
        norm=config["norm"],
        n_gen=n_gen,
        n_pop=config["n_pop"],
        n_offsprings=config["n_offsprings"],
        scale_objectives=False,
        save_history=config.get("save_history"),
        seed=config["seed"],
        n_jobs=config["system"]["n_jobs"],
        ml_scaler=scaler_augmented,
        verbose=1,
    )
    x_train_augmented_moeva = results_to_numpy_results(
        moeva.generate(x_train_augmented_candidates, 1), get_encoder_from_constraints(constraints)
    )
    np.save(x_train_augmented_moeva_path, x_train_augmented_moeva)
