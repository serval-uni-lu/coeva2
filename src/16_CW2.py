from pathlib import Path
import numpy as np
import tensorflow as tf
from art.classifiers import KerasClassifier as kc
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from utils import in_out
from art.attacks.evasion import CarliniL2Method as CW2

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
    THRESHOLD=config["threshold"]
):
    tf.compat.v1.disable_eager_execution()
    Path(ATTACK_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and Scale

    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    if INITIAL_STATE_OFFSET > - 1:
        X_initial_states = X_initial_states[
                           INITIAL_STATE_OFFSET: INITIAL_STATE_OFFSET + N_INITIAL_STATE
                           ]

    print("Attacking with {} initial states.".format(X_initial_states.shape[0]))

    # ----- Load Model

    model = tf.keras.models.load_model(MODEL_PATH)

    # ----- Attack

    kc_classifier = kc(model)
    pgd = CW2(kc_classifier, targeted=True, verbose=True, confidence=0.90)
    attacks = pgd.generate(x=X_initial_states, y=np.zeros(X_initial_states.shape[0]))
    diff = kc_classifier.predict(X_initial_states)[:, 1] - kc_classifier.predict(attacks)[:, 1]
    print("Prediction difference min: {}, avg: {}, max: {}".format(diff.min(), diff.mean(), diff.max()))
    y_pred_proba = kc_classifier.predict(attacks)
    y_attack = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)
    np.save(ATTACK_RESULTS_PATH, attacks)
    print("Success rate {}.".format((y_attack != np.ones(len(attacks))).sum()/len(attacks)))


if __name__ == "__main__":
    run()
