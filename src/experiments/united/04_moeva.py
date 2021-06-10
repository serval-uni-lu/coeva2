import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from src.attacks.moeva2.moeva2 import Moeva2
from src.experiments.united.utils import get_constraints_from_str
from src.utils import Pickler, in_out, filter_initial_states

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():

    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    save_history = True
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = filter_initial_states(
        X_initial_states, config["initial_state_offset"], config["n_initial_state"]
    )

    scaler = joblib.load(config["paths"]["ml_scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # scaler = joblib.load(config["paths"]["min_max_scaler"])

    moeva = Moeva2(
        config["paths"]["model"],
        constraints,
        problem_class=None,
        l2_ball_size=config["l2_ball_size"],
        n_gen=config["n_gen"],
        n_pop=config["n_pop"],
        n_offsprings=config["n_offsprings"],
        scale_objectives=True,
        save_history=save_history,
        seed=config["seed"],
        n_jobs=config["n_jobs"],
        ml_scaler=scaler,
        verbose=1,
    )

    attacks = moeva.generate(X_initial_states, 1)

    Pickler.save_to_file(attacks, config["paths"]["attack_results"])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


if __name__ == "__main__":
    run()
