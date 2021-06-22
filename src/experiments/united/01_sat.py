from pathlib import Path

import joblib
import numpy as np

from src.attacks.sat.sat import SatAttack
from src.experiments.united.utils import get_constraints_from_str, get_sat_constraints_from_str
from src.utils import in_out, filter_initial_states

config = in_out.get_parameters()


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )
    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    sat_constraints = get_sat_constraints_from_str(config["project_name"])
    min_max_scaler = joblib.load(config["paths"]["min_max_scaler"])

    attack = SatAttack(
        constraints,
        sat_constraints,
        min_max_scaler,
        config["thresholds"]["f2"],
        np.inf,
        n_sample=config["n_repetition"],
        verbose=1,
        n_jobs=config["n_jobs"]
    )

    x_attacks = attack.generate(x_initial)
    print(x_attacks.shape)
    np.save(config["paths"]["attack_results"], x_attacks)


if __name__ == "__main__":
    run()
