import json

import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
from datetime import datetime
from utils.in_out import save_to_file, pickle_from_file, load_from_file
from utils import in_out
from attacks.papernot_attack.problem_definition import ProblemConstraints

from attacks.papernot_attack.attack import RFAttack

config = in_out.get_parameters()


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
    experiment_id = "Papernot"

    print("running Papernot with config {} experiment id {}".format(config, experiment_id))

    model = load(MODEL_PATH)
    problem_constraints = ProblemConstraints()

    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)[
                       INITIAL_STATE_OFFSET: INITIAL_STATE_OFFSET + N_INITIAL_STATE
                       ]
    y = np.ones(len(X_initial_states))
    attack = RFAttack(model, nb_estimators=NB_ESTIMATORS, nb_iterations=NB_ITERATIONS, threshold=THRESHOLD)
    adv, rf_success_rate, l_2, l_inf = attack.generate(X_initial_states, y)

    constraints = problem_constraints.evaluate(adv)
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    respectsConstraints = 1 - constraints_violated

    isMisclassified = np.array(model.predict_proba(adv)[:, 1] < THRESHOLD).astype(
        np.int64
    )
    isBigAmount = (adv[:, 0] >= 10000).astype(np.int64)

    o3 = respectsConstraints * isMisclassified
    o4 = o3 * isBigAmount
    objectives = np.array([respectsConstraints, isMisclassified, o3, o4])
    print(objectives)
    success_rates = objectives.sum(axis=1) / objectives.shape[1]
    Path(OBJECTIVES_PATH).parent.mkdir(parents=True, exist_ok=True)

    success_rates_df = pd.DataFrame(np.array([success_rates]), columns=["o1", "o2", "o3", "o4"])
    success_rates_df.to_csv(OBJECTIVES_PATH)

    # save_to_file(objectives, OBJECTIVES_PATH)
    print(success_rates_df)
    # return adv, objectives


if __name__ == "__main__":
    run()