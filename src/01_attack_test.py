import warnings

from attacks.coeva2.lcld_constraints import LcldConstraints

warnings.simplefilter(action="ignore", category=FutureWarning)
import random
from pathlib import Path
import numpy as np
from joblib import load
from utils import Pickler, in_out
from attacks import venus_constraints, attack_multiple_input
from attacks.venus_encoder import VenusEncoder
from attacks.result_process import EfficientResult, HistoryResult
from attacks.coeva2.constraints import Constraints
from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    N_GEN=config["n_gen"],
    POP_SIZE=config["pop_size"],
    N_OFFSPRINGS=config["n_offsprings"],
    WEIGHTS=config["weights"],
    RANDOM_SEED=config["random_seed"],
    N_INITIAL_STATE=config["n_initial_state"],
    N_REPETITION=config["n_repetition"],
    ALGORITHM=config["algorithm"],
    INITIAL_STATE_OFFSET=config["initial_state_offset"],
):

    Path(ATTACK_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    save_history = False
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    classifier = Classifier(load(MODEL_PATH))
    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    X_initial_states = X_initial_states[
        INITIAL_STATE_OFFSET: INITIAL_STATE_OFFSET + N_INITIAL_STATE
    ]

    # ----- Check constraints

    constraints = LcldConstraints(amount_feature_index=config["amount_feature_index"])
    constraints.provision_constraints_min_max(config["paths"]["constraints"])
    constraints.provision_feature_constraints(config["paths"]["features"])
    constraints.check_constraints_error(X_initial_states)

    # ----- Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, N_REPETITION, axis=0)

    # Initial state loop (threaded)

    coeva2 = Coeva2(
        classifier,
        constraints,
        ALGORITHM,
        WEIGHTS,
        N_GEN,
        POP_SIZE,
        N_OFFSPRINGS,
    )

    efficient_results = coeva2.generate(X_initial_states)
    Pickler.save_to_file(efficient_results, ATTACK_RESULTS_PATH)


if __name__ == "__main__":
    run()
