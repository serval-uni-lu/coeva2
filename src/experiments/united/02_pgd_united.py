import os
import time
from itertools import combinations
from pathlib import Path
import comet_ml
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.attacks.pgd.atk import PGDTF2 as PGD
from src.attacks.pgd.auto_pgd import AutoProjectedGradientDescent
from src.attacks.pgd.classifier import TF2Classifier as kc
from src.attacks.sat.sat import SatAttack
from src.config_parser.config_parser import get_config, get_config_hash, save_config
from src.experiments.united.utils import (
    get_constraints_from_str,
    get_sat_constraints_from_str,
)
from src.utils import in_out, filter_initial_states
from src.utils.comet import init_comet
from src.utils.in_out import load_model


from src.attacks.moeva2.classifier import Classifier, ScalerClassifier
from src.attacks.objective_calculator import ObjectiveCalculator, objectives_to_dict


def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    out_dir = config["dirs"]["results"]
    config_hash = get_config_hash()
    mid_fix = f"{config['attack_name']}_{config['loss_evaluation']}"
    metrics_path = f"{out_dir}/metrics_{mid_fix}_{config_hash}.json"
    x_attacks_path = f"{out_dir}/x_attacks_{mid_fix}_{config_hash}.npy"
    x_histories_path = f"{out_dir}/x_history_{config_hash}.npy"

    # Config
    save_config(f"{out_dir}/config_{mid_fix}_")

    if os.path.exists(metrics_path):
        print(f"Configuration with hash {config_hash} already executed. Skipping")
        exit(0)

    tf.random.set_seed(config["seed"])
    experiment = None
    enable_comet = config.get("comet", True)
    if enable_comet:
        params = config
        experiment = init_comet(params)

    apply_sat = "sat" in config["loss_evaluation"]

    Path(config["dirs"]["results"]).mkdir(parents=True, exist_ok=True)

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

    model_base = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["scaler"])

    # ----- Check constraints

    # constraints.check_constraints_error(X_initial_states)

    # ----- Perform the attack

    start_time = time.time()

    if (not os.path.exists(x_attacks_path)) or (not os.path.exists(x_histories_path)):
        new_input = tf.keras.layers.Input(shape=X_initial_states.shape[1:])
        model = tf.keras.models.Model(inputs=[new_input], outputs=[model_base(new_input)])
        kc_classifier = kc(
            model,
            clip_values=(0.0, 1.0),
            input_shape=X_initial_states.shape[1:],
            loss_object=tf.keras.losses.categorical_crossentropy,
            nb_classes=2,
            constraints=constraints,
            scaler=scaler,
            experiment=experiment,
            parameters=config,
        )
        # Use only half eps if apply sat after
        per_attack_eps = config["eps"] / 2 if apply_sat else config["eps"]
        if "autopgd" in config.get("loss_evaluation"):
            attack = AutoProjectedGradientDescent(
                kc_classifier,
                constraints=constraints,
                scaler=scaler,
                experiment=experiment,
                parameters=config,
                eps=per_attack_eps - 0.000001,
                eps_step=per_attack_eps / 3,
                loss_type="cross_entropy",
                nb_random_init=config.get("nb_random", 1),
                max_iter=int(config.get("budget")),
                batch_size=X_initial_states.shape[0],
            )
        else:
            attack = PGD(
                kc_classifier,
                eps=per_attack_eps - 0.000001,
                eps_step=0.1,
                norm=config.get("norm"),
                verbose=config["system"]["verbose"] == 1,
                max_iter=int(config.get("budget")),
                num_random_init=config.get("nb_random", 0),
                batch_size=X_initial_states.shape[0],
                loss_evaluation=config.get("loss_evaluation"),
            )
        x_attacks = scaler.inverse_transform(
            attack.generate(
                x=scaler.transform(X_initial_states),
                y=to_categorical(y_initial_states),
                mask=constraints.get_mutable_mask(),
            )
        )
        mask_int = constraints.get_feature_type() != "real"
        x_attacks_int = x_attacks[:, mask_int]
        x_plus_minus = x_attacks_int - X_initial_states[:, mask_int] >= 0
        x_attacks_int[x_plus_minus] = np.floor(
            x_attacks_int[x_plus_minus]
        )
        x_attacks_int[~x_plus_minus] = np.ceil(
            x_attacks_int[~x_plus_minus]
        )
        x_attacks[:, mask_int] = x_attacks_int

        # Apply sat if needed

        if apply_sat:
            sat_constraints = get_sat_constraints_from_str(config["project_name"])
            attack = SatAttack(
                constraints,
                sat_constraints,
                scaler,
                per_attack_eps,
                np.inf,
                n_sample=1,
                verbose=1,
                n_jobs=config["system"]["n_jobs"],
            )
            x_attacks = attack.generate(X_initial_states, x_attacks)

        if config["reconstruction"]:
            important_features = np.load("./data/lcld/important_features.npy")
            combi = -sum(1 for i in combinations(range(len(important_features)), 2))
            x_attacks_l = x_attacks[:, :combi]
            print(x_attacks_l.shape)
            x_attacks = augment_data(x_attacks_l, important_features)
            print(x_attacks.shape)

        consumed_time = time.time() - start_time
        # ----- End attack

        if len(x_attacks.shape) == 2:
            x_attacks = x_attacks[:, np.newaxis, :]

        np.save(x_attacks_path, x_attacks)

        # History
        if config.get("save_history") in ["reduced", "full"]:
            history = np.swapaxes(np.array(kc_classifier.history), 0, 1)
            history = history[:, :, np.newaxis, :]
            np.save(x_histories_path, history)

    else:
        x_attacks = np.load(x_attacks_path)
        consumed_time = -1

    classifier = ScalerClassifier(config["paths"]["model"], config["paths"]["scaler"])
    thresholds = {"model": config["classification_threshold"], "distance": config["eps"]}

    objective_calc = ObjectiveCalculator(
        classifier,
        constraints,
        thresholds=thresholds,
        fun_distance_preprocess=scaler.transform,
        norm=config["norm"],
    )

    success_rate = objective_calc.get_success_rates(X_initial_states, y_initial_states, x_attacks)
    success_rate = objectives_to_dict(success_rate)

    # Metrics
    metrics = {
        "objectives": success_rate,
        "time": consumed_time,
        "config": config,
        "config_hash": config_hash,
    }

    in_out.json_to_file(metrics, metrics_path)




if __name__ == "__main__":
    config = get_config()
    run()
    # To allow the metrics to be uploaded
    # time.sleep(30)
