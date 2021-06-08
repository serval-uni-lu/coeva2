from pathlib import Path

import joblib
import numpy as np

from src.attacks.sat.sat import SatAttack
from src.examples.lcld.lcld_constraints import LcldConstraints
from src.examples.lcld.lcld_constraints_sat import (
    create_constraints as lcld_sat_constraints,
)
from src.utils import in_out, filter_initial_states

config = in_out.get_parameters()


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )
    constraints = LcldConstraints(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    min_max_scaler = joblib.load(config["paths"]["min_max_scaler"])

    attack = SatAttack(
        constraints,
        lcld_sat_constraints,
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