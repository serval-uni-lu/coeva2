import numpy as np

from src.attacks.moeva2.classifier import ScalerClassifier
from src.config_parser.config_parser import get_config


def get_prediction(y_score, threshold):
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)
    return y_pred


def run():

    for project in config.get("projects"):
        threshold = config.get("threshold")
        model_path = f"./models/{project}/baseline.model"
        scaler_path = f"./models/{project}/baseline_scaler.joblib"
        classifier = ScalerClassifier(model_path, scaler_path)

        X_train = np.load(f"./data/{project}/baseline_X_train_candidates.npy")
        y_train = np.load(f"./data/{project}/baseline_y_train_candidates.npy")
        X_test = np.load(f"./data/{project}/baseline_X_test_candidates.npy")
        y_test = np.load(f"./data/{project}/baseline_y_test_candidates.npy")
        train_index = (
            get_prediction(classifier.predict_proba(X_train), threshold) == y_train
        )
        test_index = (
            get_prediction(classifier.predict_proba(X_test), threshold) == y_test
        )
        print(f"Train: {train_index.sum}, {train_index.shape}")
        print(f"Test: {test_index.sum}, {test_index.shape}")

        indexes = {"train": train_index, "test": test_index}

        for train_test in ["train", "test"]:
            print(train_test)
            for model_name in ["baseline", "augmented"]:
                print(model_name)
                suffix = "_augmented" if model_name == "augmented" else ""
                X_path = f"./data/{project}{suffix}/{model_name}_X_{train_test}_candidates.npy"
                y_path = f"./data/{project}{suffix}/{model_name}_y_{train_test}_candidates.npy"
                X = np.load(X_path)
                y = np.load(y_path)
                X = X[indexes[train_test]]
                y = y[indexes[train_test]]
                np.save(X_path, X)
                np.save(y_path, y)
                print(X.shape)
                print(y.shape)


if __name__ == "__main__":
    config = get_config()

    run()
