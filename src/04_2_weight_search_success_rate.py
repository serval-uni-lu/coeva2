import pandas as pd
from joblib import load
import numpy as np

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.lcld_constraints import LcldConstraints
from attacks.coeva2.objective_calculator import ObjectiveCalculator
from attacks.objectives import calculate_success_rates
from attacks.venus_encoder import VenusEncoder
from utils import Pickler, in_out
from utils.in_out import pickle_from_dir

config = in_out.get_parameters()


out_columns = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "o1",
    "o2",
    "o3",
    "o4",
]


def process(results, objective_calculator):

    success_rates = objective_calculator.success_rate(results)
    return np.concatenate(
        [
            np.array(
                [
                    results[0].weight["alpha"],
                    results[0].weight["beta"],
                    results[0].weight["gamma"],
                    results[0].weight["delta"],
                ]
            ),
            success_rates,
        ]
    )


def run(
    ATTACK_RESULTS_DIR=config["dirs"]["attack_results"],
    OBJECTIVES_PATH=config["paths"]["objectives"],
    THRESHOLD=config["threshold"],
    MODEL_PATH=config["paths"]["model"],
):

    classifier = Classifier(load(config["paths"]["model"]))
    constraints = LcldConstraints(
        config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    objective_calculator = ObjectiveCalculator(
        classifier,
        constraints,
        config["threshold"],
        config["high_amount"],
        config["amount_feature_index"]
    )

    success_rates = np.array(
        pickle_from_dir(
            ATTACK_RESULTS_DIR,
            handler=lambda i, x: process(x, objective_calculator),
            n_jobs=10
        )
    )

    success_rates_df = pd.DataFrame(success_rates, columns=out_columns)
    success_rates_df.to_csv(OBJECTIVES_PATH, index=False)


if __name__ == "__main__":
    run()
