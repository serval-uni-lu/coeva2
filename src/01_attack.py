import warnings

from attacks.coeva2.lcld_constraints import LcldConstraints

import random
from pathlib import Path
import numpy as np
from joblib import load
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import Pickler, in_out, load_keras_model
from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2
from datetime import datetime
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    save_history = True
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    #classifier = load_keras_model.MDModel(config["paths"]["model"])
    keras_model = load_model(config["paths"]["model"])
    keras_model.summary()
    classifier = Classifier(keras_model)
    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = X_initial_states[
        config["initial_state_offset"] : config["initial_state_offset"]
        + config["n_initial_state"]
    ]
    constraints = LcldConstraints(
        # config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Set random seed
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

    # Initial state loop (threaded)

    coeva2 = Coeva2(
        classifier,
        constraints,
        config["algorithm"],
        config["weights"],
        config["n_gen"],
        config["pop_size"],
        config["n_offsprings"],
        save_history=save_history,
        n_jobs=config["n_jobs"]
    )

    efficient_results = coeva2.generate(X_initial_states)
    Pickler.save_to_file(efficient_results, config["paths"]["attack_results"])
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

if __name__ == "__main__":
    run()
