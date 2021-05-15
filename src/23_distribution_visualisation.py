import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from attacks.coeva2.classifier import Classifier
from src.examples.lcld.lcld_constraints import LcldConstraints
from attacks.coeva2.objective_calculator import ObjectiveCalculator
from utils import Pickler, in_out

config = in_out.get_parameters()
LOGGER = logging.getLogger()


def run():
    # train_test_data_dir = config["dirs"]["train_test_data"]
    # x_train = np.load("{}/X_train.npy".format(train_test_data_dir))[:100000]
    # y_train = np.load("{}/y_train.npy".format(train_test_data_dir))[:100000]
    # x_adv = np.load(config["paths"]["x_adv"])[:2000]
    scaler = Pickler.load_from_file(config["paths"]["scaler"])
    # x_train_s = scaler.fit_transform(x_train)

    # print(np.concatenate((x_train_s, x_adv_s)))

    model = load(config["paths"]["model"])
    classifier = Classifier(model)
    constraints = LcldConstraints(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    objective_calculator = ObjectiveCalculator(
        classifier,
        constraints,
        config["threshold"],
    )

    attack_results = Pickler.load_from_file(config["paths"]["attack_results"])
    generated = objective_calculator.get_generated(attack_results)

    for g in generated:
        for col in ["x_adv_success", "x_adv_fail"]:
            if g[col].shape[0] > 0:
                g[f"{col}_diff"] = np.abs(
                    scaler.transform(g["x_initial_state"].reshape(1, -1))
                    - scaler.transform(g[col])
                ).mean(axis=0)

    x_adv_success_diff = pd.DataFrame([
        g["x_adv_success_diff"] for g in generated if "x_adv_success_diff" in g
    ], columns=range(756))

    x_adv_fail_diff = pd.DataFrame([
        g["x_adv_fail_diff"] for g in generated if "x_adv_fail_diff" in g
    ], columns=range(756))

    intresting_col = x_adv_success_diff.max().sort_values(ascending=False)[:10].index

    x_adv_success_diff[intresting_col].boxplot()
    plt.show()
    x_adv_fail_diff[intresting_col].boxplot()
    plt.show()
    print(x_adv_success_diff.shape)


if __name__ == "__main__":
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)

    run()
