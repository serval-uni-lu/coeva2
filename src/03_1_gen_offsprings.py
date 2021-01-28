import warnings

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.coeva2 import Coeva2
from attacks.coeva2.lcld_constraints import LcldConstraints
import random
import logging
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed, load
from utils import Pickler, in_out

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.INFO)

config = in_out.get_parameters()


def run():
    Path(config["dirs"]["attack_results"]).mkdir(parents=True, exist_ok=True)

    save_history = False
    if "save_history" in config:
        save_history = config["save_history"]

    # ----- Load and create necessary objects

    classifier = Classifier(load(config["paths"]["model"]))
    X_initial_states = np.load(config["paths"]["x_candidates"])
    X_initial_states = X_initial_states[
        config["initial_state_offset"] : config["initial_state_offset"]
        + config["n_initial_state"]
    ]
    constraints = LcldConstraints(
        config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    # ----- Check constraints

    constraints.check_constraints_error(X_initial_states)

    # ----- Set random seed
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # ----- Copy the initial states n_repetition times
    X_initial_states = np.repeat(X_initial_states, config["n_repetition"], axis=0)

    list_n_offsprings = np.array(config["list_n_offsprings"])
    list_n_generation = (config["budget"] / list_n_offsprings).astype(np.int)
    list_pop_size = list_n_offsprings * 2

    for i in range(len(list_n_offsprings)):
        n_gen = list_n_generation[i]
        pop_size = list_pop_size[i]
        n_offsprings = list_n_offsprings[i]

        logging.info(
            "Parameters: {} {} {} ({}/{})".format(
                n_gen, pop_size, n_offsprings, i + 1, len(list_n_offsprings)
            )
        )

        # Initial state loop (threaded)

        coeva2 = Coeva2(
            classifier,
            constraints,
            config["algorithm"],
            config["weights"],
            n_gen,
            pop_size,
            n_offsprings,
            save_history=save_history,
            n_jobs=config["n_jobs"]
        )

        efficient_results = coeva2.generate(X_initial_states)

        Pickler.save_to_file(
            efficient_results,
            "{}/results_{}_{}.pickle".format(
                config["dirs"]["attack_results"], n_gen, config["initial_state_offset"]
            ),
        )


if __name__ == "__main__":
    run()
