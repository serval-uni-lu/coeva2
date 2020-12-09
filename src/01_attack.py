import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import random
from pathlib import Path
import numpy as np
from joblib import load
from utils import Pickler, in_out
from attacks import venus_constraints, attack_multiple_input
from attacks.venus_encoder import VenusEncoder
from attacks.result_process import EfficientResult, HistoryResult

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    SCALER_PATH=config["paths"]["scaler"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    N_GEN=config["n_gen"],
    POP_SIZE=config["pop_size"],
    N_OFFSPRINGS=config["n_offsprings"],
    WEIGHT=config["weights"],
    RANDOM_SEED=config["random_seed"],
    N_INITIAL_STATE=config["n_initial_state"],
    N_REPETITION=config["n_repetition"],
    ALGORITHM=config["algorithm"],
    INITIAL_STATE_OFFSET=config["initial_state_offset"],
):

    Path(ATTACK_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    save_history=False
    if "save_history" in config:
        save_history=config["save_history"]

    # ----- Load and create necessary objects

    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    scaler = Pickler.load_from_file(SCALER_PATH)
    encoder = VenusEncoder()
    X_initial_states = X_initial_states[
        INITIAL_STATE_OFFSET : INITIAL_STATE_OFFSET + N_INITIAL_STATE
    ]

    # ----- Check constraints

    venus_constraints.respect_constraints_or_exit(X_initial_states)

    # ----- Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, N_REPETITION, axis=0)

    # Initial state loop (threaded)

    results = attack_multiple_input.attack(
        model,
        scaler,
        encoder,
        N_GEN,
        POP_SIZE,
        N_OFFSPRINGS,
        X_initial_states,
        weight=WEIGHT,
        attack_type=ALGORITHM,
        save_history=save_history
    )

    if save_history:
        efficient_results = [HistoryResult(result) for result in results]
    else:
        efficient_results = [EfficientResult(result) for result in results]

    Pickler.save_to_file(efficient_results, ATTACK_RESULTS_PATH)


if __name__ == "__main__":
    run()
