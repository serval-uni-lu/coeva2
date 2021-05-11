import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from art.attacks.evasion import ProjectedGradientDescent as PGD
from art.classifiers import KerasClassifier as kc
from tensorflow.keras.models import load_model

from utils import in_out

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    N_INITIAL_STATE=config["n_initial_state"],
    INITIAL_STATE_OFFSET=config["initial_state_offset"],
):
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.disable_eager_execution()
    Path(ATTACK_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and Scale

    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    if N_INITIAL_STATE > -1:
        X_initial_states = X_initial_states[
            INITIAL_STATE_OFFSET : INITIAL_STATE_OFFSET + N_INITIAL_STATE
        ]
    scaler = joblib.load(config["paths"]["scaler"])
    X_initial_states = scaler.transform(X_initial_states)
    logging.info(f"Attacking with {X_initial_states.shape[0]} initial states.")

    # ----- Load Model

    model = load_model(MODEL_PATH)

    # ----- Attack

    kc_classifier = kc(
        model,
        clip_values=(0.0, 1.0),
    )
    pgd = PGD(kc_classifier, eps=0.1, targeted=True, verbose=True)
    attacks = pgd.generate(x=X_initial_states, y=np.zeros(X_initial_states.shape[0]))

    logging.info(f"{attacks.shape[0]} attacks generated")
    np.save(ATTACK_RESULTS_PATH, attacks)


if __name__ == "__main__":
    run()
