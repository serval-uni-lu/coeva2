import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from src.attacks.moeva2.classifier import ScalerClassifier
from src.attacks.objective_calculator import ObjectiveCalculator
from src.config_parser.config_parser import get_config
from src.datasets import load_dataset
from src.experiments.united.important_utils import augment_dataset
from src.experiments.united.utils import get_constraints_from_str, save_model
from src.models import load_model_architecture
from src.utils.in_out import load_model, json_to_file, json_from_file
from src.utils.ml import convert_2d_score
from sklearn.ensemble import RandomForestClassifier

def calc_n_important_features(n_features, ratio):
    a, b, c = 0.5, -0.5, -n_features * ratio
    n_important_features = np.floor(np.roots([a, b, c])[0])
    return n_important_features


def print_score(y_test, y_score, threshold=None):

    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)

    print("Test Result:\n================================================")
    print(f"AUROC: {roc_auc_score(y_test, y_score[:, 1])}")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end="")
    print(f"\tPrecision Score: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tRecall Score: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tF1 score: {f1_score(y_test, y_pred) * 100:.2f}%")
    print(f"\t\t\tMCC score: {matthews_corrcoef(y_test, y_pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}\n")


def calc_binary_metrics(y_test, y_score, threshold=None):
    print(threshold)
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        print("hello")
        y_pred = (y_score[:, 1] >= threshold).astype(np.int64)

    metrics = {
        "roc_auc_score": roc_auc_score(y_test, y_score[:, 1]),
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "recall_score": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_evaluate_model(
    project,
    model_name,
    threshold,
    scaler,
    X_train,
    X_test,
    y_train,
    y_test,
    overwrite,
    model_arch_name=None,
):
    if model_arch_name is None:
        model_arch_name = project

    scaler_path = f"./models/{project}/{model_name}_scaler.joblib"
    if not os.path.exists(scaler_path) or overwrite:
        joblib.dump(scaler, scaler_path)

    model_arch = load_model_architecture(model_arch_name)

    model_path = f"./models/{project}/{model_name}.model"
    if not os.path.exists(model_path) or overwrite:
        model = model_arch.get_trained_model(scaler.transform(X_train), y_train)
        save_model(model, model_path)
    else:
        model = load_model(model_path)

    metrics_path = f"./models/{project}/{model_name}_metrics.json"
    if not os.path.exists(metrics_path) or overwrite:
        print(X_test.shape)
        if isinstance(model, RandomForestClassifier):
            y_score = convert_2d_score(model.predict_proba(scaler.transform(X_test)))
        else:
            y_score = convert_2d_score(model.predict(scaler.transform(X_test)))
        metrics = calc_binary_metrics(y_test, y_score, threshold)
        json_to_file(metrics, metrics_path)
    else:
        metrics = json_from_file(metrics_path)

    return model, metrics, scaler_path, model_path


def save_candidates(project, model_name, attack_classes, X, y, name):
    attack_candidate_filter = np.zeros(X.shape[0]).astype(np.bool)
    for attack_class in attack_classes:
        attack_candidate_filter = attack_candidate_filter + (y == attack_class)

    X_candidates = X[attack_candidate_filter]
    y_candidates = y[attack_candidate_filter]
    print(f"Attack candidates shape {X_candidates.shape}")
    X_candidates_path = f"./data/{project}/{model_name}_X_{name}_candidates.npy"
    y_candidates_path = f"./data/{project}/{model_name}_y_{name}_candidates.npy"
    if not (os.path.exists(X_candidates_path) and os.path.exists(y_candidates_path)):
        np.save(X_candidates_path, X_candidates)
        np.save(y_candidates_path, y_candidates)
    else:
        X_candidates = np.load(X_candidates_path)
        y_candidates = np.load(y_candidates_path)
    return X_candidates, y_candidates


def run_project(project, overwrite):
    # Load
    dataset = load_dataset(project)
    X_train, X_test, y_train, y_test = dataset.get_train_test()
    scaler = dataset.get_scaler()
    threshold = config.get("classification_threshold")

    constraints = get_constraints_from_str(project)()

    # constraints.check_constraints_error(X_test)

    # constraints.check_constraints_error(X_train)

    # Normal model
    model_name = "baseline"
    model, metrics, scaler_path, model_path = train_evaluate_model(
        project,
        model_name,
        threshold,
        scaler,
        X_train,
        X_test,
        y_train,
        y_test,
        overwrite,
    )
    print(f"Threshold: {threshold}")
    print(metrics)

    attack_classes = config.get("attack_classes")
    x_train_candidates, y_train_candidates = save_candidates(
        project, model_name, attack_classes, X_train, y_train, "train"
    )
    save_candidates(project, model_name, attack_classes, X_test, y_test, "test")

    # Adversarial models

    ## Check if doable
    do_adv_training = all(
        [
            os.path.exists(x_attack["path"])
            and (not os.path.exists(f"./models/{project}/adv_{x_attack['name']}.model"))
            for x_attack in config.get("x_attacks", [])
        ]
    )
    if do_adv_training:
        # Get the adversarials:
        x_train_candidates = np.load(f"./data/{project}/baseline_X_train_candidates.npy")
        y_train_candidates = np.load(f"./data/{project}/baseline_y_train_candidates.npy")
        print(f"x_train_candidates.shape{x_train_candidates.shape}")
        classifier = ScalerClassifier(model_path, scaler_path)
        objective_calculator = ObjectiveCalculator(
            classifier,
            constraints,    
            thresholds={
                "model": threshold if threshold is not None else 0.5,
                "distance": config.get("distance"),
            },
            fun_distance_preprocess=scaler.transform,
            norm=config.get("norm"),
            n_jobs=32
        )
        x_attacks_and_i = []
        x_adv_all_i = np.arange(x_train_candidates.shape[0])
        for x_attack in config.get("x_attacks", []):
            x_attacks = np.load(x_attack["path"])
            x_adv, x_adv_i = objective_calculator.get_successful_attacks(
                x_train_candidates,
                y_train_candidates,
                x_attacks,
                preferred_metrics="misclassification",
                order="asc",
                max_inputs=1,
                return_index_success=True,
            )
            # x_adv = np.concatenate(x_adv)
            x_attacks_and_i.append((x_adv, x_adv_i))
            x_adv_all_i = np.intersect1d(x_adv_all_i, x_adv_i, return_indices=False)
            print(f"Intersection number {x_adv_all_i.shape}")

        shuffle_i = np.arange(X_train.shape[0] + x_adv_all_i.shape[0])
        rng = np.random.default_rng(42)
        rng.shuffle(shuffle_i)
        for x_attack_i, x_attack in enumerate(config.get("x_attacks", [])):
            X_train_adv = np.concatenate(
                [
                    X_train,
                    np.concatenate(
                        np.array(x_attacks_and_i[x_attack_i][0])[x_adv_all_i]
                    ),
                ]
            )
            y_train_adv = np.concatenate(
                [y_train, y_train_candidates[x_adv_all_i]]
            )

            X_train_adv = X_train_adv[shuffle_i]
            y_train_adv = y_train_adv[shuffle_i]

            model, metrics, scaler_path, model_path = train_evaluate_model(
                project,
                f"adv_{x_attack['name']}",
                threshold,
                scaler,
                X_train_adv,
                X_test,
                y_train_adv,
                y_test,
                overwrite,
            )
            print(f"Threshold: {threshold}")
            print(metrics)

    # # Random forest
    # model_name = "baseline_rf"
    # model_rf, metrics, _, _ = train_evaluate_model(
    #     project,
    #     model_name,
    #     threshold,
    #     scaler,
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     overwrite,
    #     model_arch_name=f"{project}_rf",
    # )
    # print(f"Threshold: {threshold}")
    # print(metrics)

    # Augment
    X_train_augmented_path = f"./data/{project}_augmented/X_train.npy"
    X_test_augmented_path = f"./data/{project}_augmented/X_test.npy"
    augmented_features_path = f"./data/{project}_augmented/features.csv"
    if not os.path.exists(X_train_augmented_path) or overwrite:
        X_train_augmented, X_test_augmented, features_augmented, important_features = augment_dataset(
            model,
            scaler,
            pd.read_csv(f"./data/{project}/features.csv"),
            X_train,
            y_train,
            X_test,
            ratio=0.25,
        )
        features_augmented.to_csv(augmented_features_path)
        np.save(X_train_augmented_path, X_train_augmented)
        np.save(X_test_augmented_path, X_test_augmented)
        np.save(f"./data/{project}_augmented/y_train.npy", y_train)
        np.save(f"./data/{project}_augmented/y_test.npy", y_test)
        np.save(f"./data/{project}_augmented/important_features.npy", important_features)

    dataset = load_dataset(f"{project}_augmented")
    X_train_augmented, X_test_augmented, y_train, y_test = dataset.get_train_test()
    print(
        f"X_train.shape: {X_train.shape}, X_train_augmented.shape: {X_train_augmented.shape}"
    )
    scaler = dataset.get_scaler()
    threshold = config.get("classification_threshold")

    # Augmented model

    model_name = "augmented"
    model, metrics, _, _ = train_evaluate_model(
        project,
        model_name,
        threshold,
        scaler,
        X_train_augmented,
        X_test_augmented,
        y_train,
        y_test,
        overwrite,
    )
    print(f"Threshold: {threshold}")
    print(metrics)

    save_candidates(
        f"{project}_augmented",
        model_name,
        attack_classes,
        X_train_augmented,
        y_train,
        "train",
    )
    save_candidates(
        f"{project}_augmented",
        model_name,
        attack_classes,
        X_test_augmented,
        y_test,
        "test",
    )


def run():
    overwrite = config["overwrite"]

    # Train model
    for project in config["projects"]:
        run_project(project, overwrite)


if __name__ == "__main__":
    config = get_config()

    run()
