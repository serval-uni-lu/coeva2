import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

from pathlib import Path
from joblib import dump
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import Datafilter
from utils import Pickler
from utils import in_out
from attacks import venus_constraints

config = in_out.get_parameters()


def run(
    DATASET_PATH=config["paths"]["dataset"],
    MODEL_PATH=config["paths"]["model"],
    SCALER_PATH=config["paths"]["scaler"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    TARGET=config["target"],
    MODEL_PARAMETERS=config["model_parameters"],
    THRESHOLD=config["threshold"],
    TRAIN_TEST_DATA_DIR=config["dirs"]["train_test_data"],
):
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATASET_PATH)
    y = data.pop(TARGET).to_numpy()
    X = data.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    np.save("{}/X_train.npy".format(TRAIN_TEST_DATA_DIR), X_train)
    np.save("{}/X_test.npy".format(TRAIN_TEST_DATA_DIR), X_test)
    np.save("{}/y_train.npy".format(TRAIN_TEST_DATA_DIR), y_train)
    np.save("{}/y_test.npy".format(TRAIN_TEST_DATA_DIR), y_test)

    # ----- DEFINE, TRAIN AND SAVE CLASSIFIER

    model = RandomForestClassifier(**MODEL_PARAMETERS, verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)
    model.set_params(verbose=0)
    dump(model, MODEL_PATH)

    # ----- SAVE X correctly rejected loans and respecting constraints

    X_test, y_test, y_pred = Datafilter.filter_correct_prediction(
        X_test, y_test, y_pred
    )
    X_test, y_test, y_pred = Datafilter.filter_by_target_class(
        X_test, y_test, y_pred, 1
    )
    X_test = X_test[np.random.permutation(X_test.shape[0])]

    # Removing x that violates constraints
    constraints = venus_constraints.evaluate(X_test)
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_test = X_test[(1 - constraints_violated).astype(bool)]
    np.save(X_ATTACK_CANDIDATES_PATH, X_test)

    # ----- Create and save min max scaler

    scaler = MinMaxScaler()
    scaler.fit(X)
    Pickler.save_to_file(scaler, SCALER_PATH)


if __name__ == "__main__":
    run()
