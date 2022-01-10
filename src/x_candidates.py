import numpy as np

from src.attacks.moeva2.classifier import ScalerClassifier
from src.config_parser.config_parser import get_config
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from src.utils.ml import convert_2d_score


def calc_binary_metrics(y_test, y_score, threshold=None):
    print(threshold)
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        print("hello")
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)

    metrics = {
        # "roc_auc_score": roc_auc_score(y_test, y_score[:, 1]),
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def get_prediction(y_score, threshold):
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)
    return y_pred


def print_score(y_test, y_score, threshold=None):

    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)

    print("Test Result:\n================================================")
    # print(f"AUROC: {roc_auc_score(y_test, y_score[:, 1])}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end="")
    print(f"\tPrecision Score: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tRecall Score: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tF1 score: {f1_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tMCC score: {matthews_corrcoef(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}\n")


def run():

    for project in config.get("projects"):
        print(f"-----{project}")
        threshold = config.get("threshold")
        model_path = f"./models/{project}/baseline.model"
        scaler_path = f"./models/{project}/baseline_scaler.joblib"
        classifier = ScalerClassifier(model_path, scaler_path)

        X_train = np.load(f"./data/{project}/baseline_X_train_candidates.npy")
        print(X_train.shape)
        y_train = np.load(f"./data/{project}/baseline_y_train_candidates.npy")
        X_test = np.load(f"./data/{project}/baseline_X_test_candidates.npy")
        y_test = np.load(f"./data/{project}/baseline_y_test_candidates.npy")
        train_index = (
            get_prediction(classifier.predict_proba(X_train), threshold) == y_train
        )
        test_index = (
            get_prediction(classifier.predict_proba(X_test), threshold) == y_test
        )
        print(threshold)
        # print_score(np.load(f"./data/{project}/y_test.npy"), classifier.predict_proba(np.load(f"./data/{project}/X_test.npy")), threshold)
        print(calc_binary_metrics(np.load(f"./data/{project}/y_test.npy"), convert_2d_score(classifier.predict_proba(np.load(f"./data/{project}/X_test.npy"))), threshold))
        print(classifier.scaler)

        print(f"Train: {train_index.sum()}, {train_index.shape}")
        print(f"Test: {test_index.sum()}, {test_index.shape}")
        print(f"y {y_train.sum()/ y_train.shape[0]}")

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
