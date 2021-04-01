import json

import pandas as pd
import numpy as np
from joblib import load, Parallel, delayed
from pathlib import Path
from datetime import datetime
from utils.in_out import save_to_file, pickle_from_file, load_from_file
from utils import in_out
from attacks.papernot_attack.problem_definition import ProblemConstraints
import copy
import logging
from tqdm import tqdm

from attacks.papernot_attack.attack import RFAttack

config = in_out.get_parameters()


def unique_attack(model, nb_estimators, nb_iterations, threshold, X, y):
    model = copy.deepcopy(model)
    attack = RFAttack(model, nb_estimators=nb_estimators, nb_iterations=nb_iterations, threshold=threshold)
    adv, _, _, _ = attack.generate(np.array([X]), np.array([y]))
    return adv


def run(
        MODEL_PATH=config["paths"]["model"],
        X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
        ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
        N_INITIAL_STATE=config["n_initial_state"],
        INITIAL_STATE_OFFSET=config["initial_state_offset"],
        THRESHOLD=config["threshold"],
        NB_ESTIMATORS=config["nb_estimators"],
        NB_ITERATIONS=config["nb_iterations"],
        OBJECTIVES_PATH=config["paths"]["objectives"],
):
    logging.basicConfig(level=logging.INFO)
    experiment_id = "Papernot"

    # ----- Load and Scale

    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)
    if INITIAL_STATE_OFFSET > - 1:
        X_initial_states = X_initial_states[
                           INITIAL_STATE_OFFSET: INITIAL_STATE_OFFSET + N_INITIAL_STATE
                           ]

    logging.info(f"Attacking with {X_initial_states.shape[0]} initial states.")

    # ----- Load Model

    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)    
    
    y = np.ones(len(X_initial_states))

    # attack = RFAttack(model, nb_estimators=NB_ESTIMATORS, nb_iterations=NB_ITERATIONS, threshold=THRESHOLD)
    # attacks, _, _, _ = attack.generate(X_initial_states, y)

    # ----- Attack
    attacks = Parallel(n_jobs=-1, verbose=0)(
        delayed(unique_attack)(
            model,
            NB_ESTIMATORS,
            NB_ITERATIONS,
            THRESHOLD,
            initial_state,
            y[index]

        ) for index, initial_state in tqdm(enumerate(X_initial_states), total=X_initial_states.shape[0])
    )
    attacks = np.array(attacks)[:, 0, :]
    logging.info(f"{attacks.shape[0]} attacks generated")
    np.save(ATTACK_RESULTS_PATH, attacks)


if __name__ == "__main__":
    run()