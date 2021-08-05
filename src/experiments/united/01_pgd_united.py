import time
from datetime import datetime
from pathlib import Path
import comet_ml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.attacks.pgd.atk import PGDTF2 as PGD
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


from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator


def run():
    experiment = None
    enable_comet = config.get("comet", True)
    if enable_comet:
        params = config
        experiment = init_comet(params)

    apply_sat = "sat" in config["strategy"]

    Path(config["dirs"]["results"]).mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    x_initial = np.load(config["paths"]["x_candidates"])

    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )

    initial_shape = x_initial.shape[1:]

    model_base = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["ml_scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(x_initial)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    new_input = tf.keras.layers.Input(shape=initial_shape)
    model = tf.keras.models.Model(inputs=[new_input], outputs=[model_base(new_input)])

    kc_classifier = kc(
        model,
        clip_values=(0.0, 1.0),
        input_shape=initial_shape,
        loss_object=tf.keras.losses.binary_crossentropy,
        nb_classes=2,
        constraints=constraints,
        scaler=scaler,
        experiment=experiment,
        parameters=config,
    )
    # Use only half eps if apply sat after
    per_attack_eps = config["eps"] / 2 if apply_sat else config["eps"]
    pgd = PGD(
        kc_classifier,
        eps=per_attack_eps - 0.000001,
        eps_step=config["eps"] / 500000,
        norm=config.get("norm"),
        verbose=True,
        max_iter=config.get("nb_iter"),
        num_random_init=config.get("nb_random", 0),
        batch_size=x_initial.shape[0],
        loss_evaluation=config.get("loss_evaluation")
    )
    x_attacks = scaler.inverse_transform(
        pgd.generate(
            x=scaler.transform(x_initial),
            mask=constraints.get_mutable_mask(),
        )
    )
    mask_int = constraints.get_feature_type() != "real"
    x_attacks[:, mask_int] = np.rint(x_attacks[:, mask_int])

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
            n_jobs=config["n_jobs"],
        )
        x_attacks = attack.generate(x_initial, x_attacks)

    if len(x_attacks.shape) == 2:
        x_attacks = x_attacks[:, np.newaxis, :]
    classifier = Classifier(model_base)
    threholds = {"f1": config["misclassification_threshold"], "f2": config["eps"]}
    objective_calc = ObjectiveCalculator(
        classifier,
        constraints,
        minimize_class=1,
        thresholds=threholds,
        min_max_scaler=scaler,
        ml_scaler=scaler,
    )

    success_rate_df = objective_calc.success_rate_3d_df(x_initial, x_attacks)
    print(success_rate_df)

    for c, v in zip(success_rate_df.columns, success_rate_df.values[0]):
        experiment.log_metric(c, v)

    # Save
    config_hash = get_config_hash()
    out_dir = config["dirs"]["results"]
    mid_fix = f"{config['attack_name']}_{config['loss_evaluation']}"
    # X_attacks

    x_attacks_path = f"{out_dir}/x_attacks_{mid_fix}_{config_hash}.npy"
    np.save(x_attacks_path, x_attacks)
    experiment.log_asset(x_attacks_path)

    # Metrics

    success_rate_df.to_csv(
        f"{out_dir}/success_rate_{mid_fix}_{config_hash}.csv", index=False
    )

    # Config
    save_config(f"{out_dir}/config_{mid_fix}_")


if __name__ == "__main__":
    config = get_config()
    run()
    # To allow the metrics to be uploaded
    time.sleep(30)
