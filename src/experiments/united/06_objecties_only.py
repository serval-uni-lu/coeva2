import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from src.attacks.moeva2.classifier import ScalerClassifier
from src.attacks.objective_calculator import ObjectiveCalculator, objectives_to_dict
from src.config_parser.config_parser import get_config, get_config_hash, save_config
from src.experiments.united.utils import get_constraints_from_str
from src.utils import filter_initial_states, timing, in_out

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# config = in_out.get_parameters()


@timing
def run():
    # Do not use GPU (with parallelization)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Get config hash
    config_hash = get_config_hash()

    # Print
    print(f"Config hash:{config_hash}")
    print(f"Config {config}")

    out_dir = config["dirs"]["results"]
    mid_fix = f"{config['attack_name']}"
    config_pre_path = f"{out_dir}/config_{mid_fix}_"
    config_path = f"{config_pre_path}{get_config_hash()}.yaml"

    if os.path.exists(config_path):
        print(f"Configuration with hash {config_hash} already executed. Skipping")
        exit(0)

    Path(config["dirs"]["results"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    constraints = get_constraints_from_str(config["project_name"])()

    X_initial_states = np.load(config["paths"]["x_input"])
    y_initial_states = np.load(config["paths"]["y_input"])
    X_initial_states = filter_initial_states(
        X_initial_states, config["input_offset"], config["n_input"]
    )
    y_initial_states = filter_initial_states(
        y_initial_states, config["input_offset"], config["n_input"]
    )

    scaler = joblib.load(config["paths"]["scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Copy the initial states n_repetition times

    n_gen = config["budget"]
    start_time = time.time()

    classifier_path = config["paths"]["model"]
    scaler_path = config["paths"]["scaler"]
    classifier = ScalerClassifier(classifier_path, scaler_path)

    consumed_time = time.time() - start_time
    print(f"Execution in {consumed_time}s. Saving...")

    x_attacks = np.load(f"{out_dir}/x_attacks_{mid_fix}_{config_hash}.npy")

    objective_lists = []
    for eps in config["eps_list"]:
        thresholds = {"model": config["classification_threshold"], "distance": eps}
        objective_calc = ObjectiveCalculator(
            classifier,
            constraints,
            thresholds=thresholds,
            fun_distance_preprocess=scaler.transform,
            norm=config["norm"],
        )
        success_rate = objective_calc.get_success_rates(
            X_initial_states, y_initial_states, x_attacks
        )
        success_rate = objectives_to_dict(success_rate)
        objective_lists.append(success_rate)

    # metrics
    metrics = {
        "objectives": objective_lists,
        "time": consumed_time,
        "config": config,
        "config_hash": config_hash,
    }
    metrics_path = f"{out_dir}/metrics_{mid_fix}_{config_hash}.json"
    in_out.json_to_file(metrics, metrics_path)

    # Config

    save_config(f"{config_pre_path}")
    print("Done.")


if __name__ == "__main__":
    config = get_config()
    run()
