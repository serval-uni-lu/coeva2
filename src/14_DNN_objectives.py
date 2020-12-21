from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef

from attacks import venus_constraints
from utils import in_out, Pickler
import pandas as pd

config = in_out.get_parameters()

def print_score(label, prediction):
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end='')
    print(f"\tPrecision Score: {precision_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tRecall Score: {recall_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tF1 score: {f1_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tMCC score: {matthews_corrcoef(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")


def run(
    MODEL_PATH=config["paths"]["model"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    N_INITIAL_STATE=config["n_initial_state"],
    INITIAL_STATE_OFFSET=config["initial_state_offset"],
    THRESHOLD=config["threshold"],
    SCALER_PATH=config["paths"]["scaler"],
    OBJECTIVES_PATH=config["paths"]["objectives"],

):
    tf.compat.v1.disable_eager_execution()
    Path(ATTACK_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and Scale

    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    X_initial_states = X_initial_states[
                       INITIAL_STATE_OFFSET: INITIAL_STATE_OFFSET + N_INITIAL_STATE
                       ]
    X_attacks = np.load(ATTACK_RESULTS_PATH)
    scaler = Pickler.load_from_file(SCALER_PATH)


    # ----- Load Model

    model = tf.keras.models.load_model(MODEL_PATH)

    y_pred_proba = model.predict(X_attacks)
    # Rejected
    y_attack = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)
    # Index of non rejected
    index_success = (y_attack != np.ones(len(X_attacks)))
    gross_success_rate = (index_success.sum() / len(X_attacks))

    X_attacks_success = X_attacks[index_success]
    constraints = venus_constraints.evaluate(scaler.inverse_transform(X_attacks_success))
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_attacks_net_success = X_attacks_success[(1 - constraints_violated).astype(bool)]

    net_success_rate = (len(X_attacks_net_success) / len(X_attacks))

    distances = np.array([np.linalg.norm(X_attacks[i]-X_initial_states[i]) for i in range(X_attacks.shape[0])])
    distance_mean = distances.mean()
    print(X_attacks.shape, X_initial_states.shape)

    objectives = {
        # "n_sample": X_initial_states.shape[0],
        "gross_success_rate": np.array([gross_success_rate]),
        "real_success_rate": np.array([net_success_rate]),
        # "L2_distance": distance_mean
    }
    history_df = pd.DataFrame.from_dict(objectives)
    history_df.to_csv(OBJECTIVES_PATH)
    print(objectives)


if __name__ == "__main__":
    run()
