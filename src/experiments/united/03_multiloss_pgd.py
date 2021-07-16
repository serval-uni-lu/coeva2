"""

-Récuperer les boundaries du scaler

constraints.get_feature_min_max(dynamic_input=dynamic_input)

"""

from comet_ml import Experiment

import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from art.attacks.evasion import ProjectedGradientDescent as PGD
from src.attacks.pgd.classifier import TF2Classifier as kc

from src.experiments.united.utils import get_constraints_from_str
from src.utils import in_out, filter_initial_states
from src.utils.in_out import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():

    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = filter_initial_states(
        X_initial_states, config["initial_state_offset"], config["n_initial_state"]
    )
    initial_shape = X_initial_states.shape[1:]

    model_base = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["ml_scaler"])

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Copy the initial states n_repetition times
    # X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

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
        constraints = constraints,
        scaler = scaler
    )
    pgd = PGD(
        kc_classifier,
        eps=config["thresholds"]["f2"]-0.000001,
        eps_step=config["thresholds"]["f2"] / 3,
        norm=config["norm"],
        verbose=True,
    )
    X_initial_states = scaler.transform(X_initial_states)
    attacks = pgd.generate(
        x=X_initial_states,
        mask=constraints.get_mutable_mask(),
    )

    attacks = scaler.inverse_transform(attacks)
    np.save(config["paths"]["attack_results"], attacks)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


if __name__ == "__main__":
    run()
