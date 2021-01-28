import warnings

from attacks.coeva2.lcld_constraints import LcldConstraints

import random
from pathlib import Path
import numpy as np
from joblib import load
from utils import Pickler, in_out
from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2

warnings.simplefilter(action="ignore", category=FutureWarning)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    save_history = False
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    classifier = Classifier(load(config["paths"]["model"]))
    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = X_initial_states[
        config["initial_state_offset"] : config["initial_state_offset"]
        + config["n_initial_state"]
    ]
    constraints = LcldConstraints(
        config["amount_feature_index"],
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
    )

    efficient_results = coeva2.generate(X_initial_states)
    Pickler.save_to_file(efficient_results, config["paths"]["attack_results"])


if __name__ == "__main__":
    run()
