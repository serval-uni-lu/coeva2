from pathlib import Path

import joblib
import numpy as np

from src.attacks.sat.sat import SatAttack
from src.examples.botnet.botnet_constraints import BotnetConstraints
from src.utils import in_out, filter_initial_states
from src.examples.botnet.botnet_constraints_sat import (
    create_constraints as botnet_sat_constraints,
)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )
    constraints = BotnetConstraints(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    min_max_scaler = joblib.load(config["paths"]["min_max_scaler"])

    attack = SatAttack(
        constraints,
        botnet_sat_constraints,
        min_max_scaler,
        config["thresholds"]["f2"],
        np.inf,
        n_sample=config["n_repetition"],
        verbose=1,
    )

    x_pgd = np.load(config["paths"]["x_pgd"])

    x_attacks = attack.generate(x_initial, x_pgd)
    print(x_attacks.shape)
    np.save(config["paths"]["attack_results"], x_attacks)


if __name__ == "__main__":
    run()
