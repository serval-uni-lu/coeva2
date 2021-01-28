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
    "n_gen",
    "pop_size",
    "n_offsprings",
    "o1",
    "o2",
    "o3",
    "o4",
]


def process(results, objective_calculator):
    success_rates = objective_calculator.success_rate(results)
    return np.concatenate(
        [
            np.array([results[0].n_gen, results[0].pop_size, results[0].n_offsprings]),
            success_rates,
        ]
    )


def run():

    # Load objects
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
            config["dirs"]["attack_results"],
            handler=lambda i, x: process(x, objective_calculator),
        )
    )

    success_rates_df = pd.DataFrame(success_rates, columns=out_columns)
    success_rates_df.to_csv(config["paths"]["objectives"], index=False)


if __name__ == "__main__":
    run()
