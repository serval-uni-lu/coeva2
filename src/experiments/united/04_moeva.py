import warnings
import time
from pathlib import Path

import joblib
import numpy as np

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.feature_encoder import get_encoder_from_constraints
from src.attacks.moeva2.moeva2 import Moeva2
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.attacks.moeva2.utils import results_to_numpy_results, results_to_history
from src.config_parser.config_parser import get_config, get_config_hash, save_config
from src.experiments.united.utils import get_constraints_from_str
from src.utils import Pickler, filter_initial_states, timing, in_out
from src.utils.in_out import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# config = in_out.get_parameters()


@timing
def run():

    Path(config['dirs']['results']).parent.mkdir(parents=True, exist_ok=True)

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
    # X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

    n_gen = int(np.floor(config["budget"] / config["n_offsprings"]))
    start_time = time.time()
    moeva = Moeva2(
        config["paths"]["model"],
        constraints,
        problem_class=None,
        l2_ball_size=0.,
        norm=config["norm"],
        n_gen=n_gen,
        n_pop=config["n_pop"],
        n_offsprings=config["n_offsprings"],
        scale_objectives=True,
        save_history=save_history,
        seed=config["seed"],
        n_jobs=config["system"]["n_jobs"],
        ml_scaler=scaler,
        verbose=1,
    )
    attacks = moeva.generate(X_initial_states, 1)
    consumed_time = time.time() - start_time
    # Save
    config_hash = get_config_hash()
    out_dir = config['dirs']['results']
    # Legacy
    Pickler.save_to_file(attacks, f"{out_dir}/results_{config_hash}.npy")

    # Attacks crafted
    x_attacks = results_to_numpy_results(
        attacks, get_encoder_from_constraints(constraints)
    )
    np.save(f"{out_dir}/x_attacks_{config_hash}.npy", x_attacks)

    # History
    if save_history:
        x_histories = results_to_history(attacks)
        np.save(f"{out_dir}/x_history_{config_hash}.npy", x_histories)


    objective_lists = []
    for eps in config["eps_list"]:
        threholds = {"f1": config["misclassification_threshold"], "f2": eps}
        classifier = Classifier(load_model(config["paths"]["model"]))
        objective_calc = ObjectiveCalculator(
            classifier,
            constraints,
            minimize_class=1,
            thresholds=threholds,
            min_max_scaler=scaler,
            ml_scaler=scaler,
        )
        success_rate_df = objective_calc.success_rate_3d_df(X_initial_states, x_attacks)
        objective_lists.append(success_rate_df.to_dict(orient="records")[0])

    metrics = {
        "objectives_list": objective_lists,
        "time": consumed_time,
        "config": config,
        "config_hash": config_hash,
    }
    mid_fix = f"{config['attack_name']}"
    in_out.json_to_file(metrics, f"{out_dir}/metrics_{mid_fix}_{config_hash}.json")

    # Config
    save_config(f"{out_dir}/config_")


if __name__ == "__main__":
    config = get_config()
    run()
