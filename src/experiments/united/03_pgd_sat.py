import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from art.attacks.evasion import ProjectedGradientDescent as PGD
from art.classifiers import KerasClassifier as kc

from src.attacks.sat.sat import SatAttack
from src.experiments.united.utils import get_constraints_from_str, get_sat_constraints_from_str
from src.utils import in_out, filter_initial_states
from src.utils.in_out import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():

    tf.compat.v1.disable_eager_execution()
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    sat_constraints = get_sat_constraints_from_str(config["project_name"])


    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = filter_initial_states(
        X_initial_states, config["initial_state_offset"], config["n_initial_state"]
    )

    model = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["ml_scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Copy the initial states n_repetition times
    # X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # PGD Part
    kc_classifier = kc(
        model,
        clip_values=(0.0, 1.0),
    )
    eps_half= config["thresholds"]["f2"]/2
    max_iter_pgd = 100
    pgd = PGD(
        kc_classifier,
        eps=eps_half-0.000001,
        eps_step=config["thresholds"]["f2"] / 3,
        norm=config["norm"],
        verbose=True,
        max_iter=max_iter_pgd,
        batch_size=X_initial_states.shape[0]
    )
    X_initial_states = scaler.transform(X_initial_states)
    attacks = pgd.generate(
        x=X_initial_states,
        mask=constraints.get_mutable_mask(),
    )

    attacks = scaler.inverse_transform(attacks)

    print(attacks.shape)

    attack = SatAttack(
        constraints,
        sat_constraints,
        scaler,
        eps_half,
        np.inf,
        n_sample=config["n_repetition"]/max_iter_pgd,
        verbose=1,
        n_jobs=config["n_jobs"]
    )

    x_attacks = attack.generate(attacks, attacks)
    print(x_attacks.shape)
    np.save(config["paths"]["attack_results"], x_attacks)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


if __name__ == "__main__":
    run()
