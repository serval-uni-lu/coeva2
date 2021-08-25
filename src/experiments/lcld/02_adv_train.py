import os

import joblib
import numpy as np
import pandas as pd

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.config_parser.config_parser import get_config
from src.experiments.lcld.model import train_model, print_score
from src.experiments.united.utils import get_constraints_from_str
from src.utils import filter_initial_states
from src.utils.in_out import load_model

np.random.seed(205)
import tensorflow as tf

tf.random.set_seed(206)

from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ----- CONFIG
config = get_config()
project_name = "lcld"
thresholds = {"f1": 0.25, "f2": 0.05}
moeva_path = (
    "./out/attacks/lcld/l2_last/x_attacks_moeva_8a9486b2f7f945227318a86d56b64121.npy"
)
gradient_path = (
    "./out/attacks/lcld/l2_last/x_attacks_pgd_constraints+"
    "flip+adaptive_eps_step+repair_3a5abe91f37272772fa852184972ef29.npy"
)
norm = 2
candidates_used = [0, 4000]
max_input_for_retrain = 791

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
y_pred = (y_proba >= thresholds["f1"]).astype(int)
print_score(y_test, y_pred)

# ----- LOAD CANDIDATES
x_candidates_path = f"./data/{project_name}/x_candidates_common.npy"
x_candidates = np.load(x_candidates_path)

# ----- LOAD Adversarials
x_moeva = np.load(moeva_path)
x_gradient = np.load(gradient_path)

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
    thresholds=thresholds,
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

    # ---- RETRIEVE ADV
    if os.path.exists(x_train_adv_path) and os.path.exists(y_train_adv_path):
        x_train_adv = np.load(x_train_adv_path)
        y_train_adv = np.load(y_train_adv_path)
    else:
        x_attacks_adv = objective_calc.get_successful_attacks(
            x_candidates_used,
            x_moeva,
            preferred_metrics="distance",
            order="asc",
            max_inputs=-1,
        )[:max_input_for_retrain]
        # ---- AUGMENT DATA
        repeat = int(np.unique(y_train, return_counts=True)[1][1] / max_input_for_retrain)
        x_attacks_adv = np.repeat(x_attacks_adv, repeat, axis=0)
        x_train_adv = np.concatenate((x_train, x_attacks_adv), axis=0)
        y_train_adv = np.concatenate(
            (y_train, np.ones(x_attacks_adv.shape[0])), axis=0
        )
        np.save(x_train_adv_path, x_train_adv)
        np.save(y_train_adv_path, y_train_adv)

    # ----- TRAIN MODEL
    model_augmented_path = f"./models/{project_name}/nn_{attack_name}.model"
    if os.path.exists(model_augmented_path):
        print(f"{model_augmented_path} exists loading...")
        model_augmented = load_model(model_augmented_path)
    else:
        model_augmented = train_model(scaler.transform(x_train_adv), y_train_adv)
        tf.keras.models.save_model(
            model_augmented,
            model_augmented_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
        )

    # ----- RETURN PREDICTIONS
    y_proba = model_augmented.predict_proba(scaler.transform(x_test)).reshape(-1)
    y_pred_augmented = (y_proba >= thresholds["f1"]).astype(int)
    print_score(y_test, y_pred_augmented)
    return model_augmented


models = [
    create_adv_model(attack, name)
    for attack, name in [(x_moeva, "moeva"), (x_gradient, "gradient")]
] + [model]

x_new_candidates_path = f"./data/{project_name}/x_candidates_common_rq3.npy"

if os.path.exists(x_new_candidates_path):
    x_new_candidates = np.load(x_new_candidates_path)
else:
    x_new_candidates = filter_initial_states(x_candidates, candidates_used[1], x_candidates.shape[0] - candidates_used[1])
    y_preds = [
        (
            (m.predict_proba(scaler.transform(x_new_candidates)).reshape(-1))
            >= thresholds["f1"]
        ).astype(int)
        for m in models
    ]
    y_preds = np.column_stack(y_preds)
    x_new_candidates_index = np.min(y_preds, axis=1)
    x_new_candidates = x_new_candidates[x_new_candidates_index]
    print(x_new_candidates.shape)
    np.save(x_new_candidates_path, x_new_candidates)
