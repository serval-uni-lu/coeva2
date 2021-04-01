import warnings
import sys

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2
from attacks.coeva2.lcld_constraints import LcldConstraints

warnings.simplefilter(action="ignore", category=FutureWarning)


import random
import logging
from pathlib import Path

import numpy as np
from joblib import load

from utils import Pickler, in_out


logging.getLogger().setLevel(logging.INFO)

config = in_out.get_parameters()


def run():
    Path(config["dirs"]["attack_results"]).mkdir(parents=True, exist_ok=True)

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

    # ----- Add shift
    for i in range(config["weight_shift"]):
        weights = np.random.uniform(config["weight_min"], config["weight_max"], 4)

    for i in range(config["n_weights"]):

        weights = np.random.uniform(config["weight_min"], config["weight_max"], 4)
        weights = {
            "alpha": weights[0],
            "beta": weights[1],
            "gamma": weights[2],
            "delta": weights[3],
        }

        logging.info("Parameters: {} ({}/{})".format(weights, i + 1, config["n_weights"]))

        # Initial state loop (threaded)
        coeva2 = Coeva2(
            classifier,
            constraints,
            config["algorithm"],
            weights,
            config["n_gen"],
            config["pop_size"],
            config["n_offsprings"],
            save_history=save_history,
            n_jobs=config["n_jobs"],
        )

        efficient_results = coeva2.generate(X_initial_states)

        Pickler.save_to_file(
            efficient_results,
            "{}/results_{}_{}.pickle".format(
                config["dirs"]["attack_results"], config["weight_shift"], i
            ),
        )


if __name__ == "__main__":
    run()
