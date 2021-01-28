import warnings
import sys

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2

warnings.simplefilter(action="ignore", category=FutureWarning)

from attacks import attack_multiple_input
from attacks.result_process import EfficientResult

import random
import logging
from pathlib import Path

import numpy as np
from joblib import load

from utils import Pickler, in_out

from attacks import venus_constraints
from attacks.venus_encoder import VenusEncoder

logging.getLogger().setLevel(logging.INFO)

config = in_out.get_parameters()
config["shift"] = int(sys.argv[2])

def run(
    ATTACK_RESULTS_DIR=config["dirs"]["attack_results"],
    ALGORITHM=config["algorithm"],
    N_GEN=config["n_gen"],
    POP_SIZE=config["pop_size"],
    N_OFFSPRINGS=config["n_offsprings"],
    N_WEIGHTS=config["n_weights"],
    WEIGHT_MIN=config["weight_min"],
    WEIGHT_MAX=config["weight_max"],
):
    Path(config["dirs"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    save_history = False
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    classifier = Classifier(load(config["paths"]["model"]))
    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = X_initial_states[
                       config["initial_state_offset"]: config["initial_state_offset"]
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
    for i in range(config["shift"]):
        weights = np.random.uniform(config["weight_min"], config["weight_max"], 4)

    for i in range(N_WEIGHTS):

        weights = np.random.uniform(config["weight_min"], config["weight_max"], 4)
        weights = {
            "alpha": weights[0],
            "beta": weights[1],
            "gamma": weights[2],
            "delta": weights[3],
        }

        logging.info("Parameters: {} ({}/{})".format(weights, i + 1, N_WEIGHTS))

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
        )

        efficient_results = coeva2.generate(X_initial_states)

        Pickler.save_to_file(
            efficient_results,
            "{}/results_{}_{}.pickle".format(ATTACK_RESULTS_DIR, config["shift"], i),
        )


if __name__ == "__main__":
    run()
