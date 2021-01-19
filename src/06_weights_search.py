import warnings
import sys

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
    MODEL_PATH=config["paths"]["model"],
    SCALER_PATH=config["paths"]["scaler"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    ATTACK_RESULTS_DIR=config["dirs"]["attack_results"],
    RANDOM_SEED=config["random_seed"],
    N_REPETITION=config["n_repetition"],
    ALGORITHM=config["algorithm"],
    N_INITIAL_STATE=config["n_initial_state"],
    N_GEN=config["n_gen"],
    POP_SIZE=config["pop_size"],
    N_OFFSPRINGS=config["n_offsprings"],
    N_WEIGHTS=config["n_weights"],
    WEIGHT_MIN=config["weight_min"],
    WEIGHT_MAX=config["weight_max"],
):

    Path(ATTACK_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    scaler = Pickler.load_from_file(SCALER_PATH)
    encoder = VenusEncoder()
    X_initial_states = X_initial_states[:N_INITIAL_STATE]

    # ----- Check constraints

    venus_constraints.respect_constraints_or_exit(X_initial_states)

    # ----- Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, N_REPETITION, axis=0)

    # ----- Add shift
    for i in range(config["shift"]):
        weight = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, 4)

    for i in range(N_WEIGHTS):

        weight = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, 4)
        weight = {
            "alpha": weight[0],
            "beta": weight[1],
            "gamma": weight[2],
            "delta": weight[3],
        }

        logging.info("Parameters: {} ({}/{})".format(weight, i + 1, N_WEIGHTS))

        # Initial state loop (threaded)
        results = attack_multiple_input.attack(
            model,
            scaler,
            encoder,
            N_GEN,
            POP_SIZE,
            N_OFFSPRINGS,
            X_initial_states,
            weight=weight,
            attack_type=ALGORITHM,
        )

        efficient_results = [EfficientResult(result) for result in results]

        Pickler.save_to_file(
            efficient_results,
            "{}/results_{}_{}.pickle".format(ATTACK_RESULTS_DIR, config["shift"], i),
        )


if __name__ == "__main__":
    run()
