from pathlib import Path
import random

from joblib import load

from attacks.iterative_papernot.attack import RFAttack
from utils import in_out
import numpy as np
import pandas as pd
from attacks.venus_constraints import evaluate

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    RANDOM_SEED=config["random_seed"],
    N_INITIAL_STATE=config["n_initial_state"],
    N_REPETITION=config["n_repetition"],
    THRESHOLD=config["threshold"],
    OBJECTIVES_PATH=config["paths"]["objectives"],
):

    Path(OBJECTIVES_PATH).parent.mkdir(parents=True, exist_ok=True)

    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
    X_initial_states = np.load(X_ATTACK_CANDIDATES_PATH)

    # ----- Set random seed

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    attack = RFAttack(
        model,
        nb_estimators=model.n_estimators,
        nb_iterations=N_REPETITION,
        threshold=THRESHOLD,
    )
    adv, rf_success_rate, l_2, l_inf = attack.generate(
        X_initial_states[:N_INITIAL_STATE], np.zeros(N_INITIAL_STATE)
    )

    constraints = evaluate(adv)
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
    objectives = objectives.sum(axis=1)
    objectives = (objectives > 0).astype(np.int64)
    success_rates = objectives / N_INITIAL_STATE

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(success_rates.reshape([1, -1]), columns=columns,)
    success_rate_df.to_csv(OBJECTIVES_PATH, index=False)


if __name__ == "__main__":
    run()
