import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.utils import get_scaler_from_norm
from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.examples.lcld.lcld_constraints import LcldConstraints
from src.utils import Pickler, in_out

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()
from src.utils.in_out import load_model

AVOID_ZERO = 0.00000001


def run():
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    efficient_results = Pickler.load_from_file(config["paths"]["attack_results"])

    histories = [
        [g["F"].tolist() for i, g in enumerate(r.history) if i > 0]
        for r in efficient_results
    ]
    histories = np.array(histories)

    # Objective scalers (Compute only once)

    f2_scaler = get_scaler_from_norm(
        config["norm"], efficient_results[0].initial_state.shape[0]
    )

    shape = histories[..., 0].shape
    histories[..., 1] = f2_scaler.inverse_transform(histories[..., 1].reshape(-1, 1)).reshape(shape)
    # histories[...,2] = histories[...,2] * 255

    working = np.min(histories, axis=2)
    working = np.min(working, axis=0)
    print(working.shape)
    for i in range(working.shape[1])[:1]:
        plt.plot(working[:, i], label=f"{i}")

    # plt.ylim(bottom=-1.0)
    # plt.plot(working[:, 0], label=f"{0}")

    # plot thresholds
    # for key in config["thresholds"]:
    #     plt.plot(
    #         np.full(working.shape[0], config["thresholds"][key]),
    #         label=f"Threshold {key}",
    #     )

    plt.yscale("linear")
    plt.legend()
    plt.savefig("plot_malware_softmax.pdf")
    # constraints = LcldConstraints(
    #     config["paths"]["features"],
    #     config["paths"]["constraints"],
    # )
    #
    # graphs = np.mean(hi)
    #
    # classifier = Classifier(load_model(config["paths"]["model"]))
    #
    # scaler = joblib.load(config["paths"]["min_max_scaler"])
    # objective_calc = ObjectiveCalculator(
    #     classifier,
    #     constraints,
    #     minimize_class=1,
    #     thresholds=config["thresholds"],
    #     min_max_scaler=scaler,
    #     ml_scaler=scaler,
    # )
    # success_rates = objective_calc.success_rate_genetic(efficient_results)
    #
    # columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    # success_rate_df = pd.DataFrame(
    #     success_rates.reshape([1, -1]),
    #     columns=columns,
    # )
    # success_rate_df.to_csv(config["paths"]["objectives"], index=False)
    # print(success_rate_df)


if __name__ == "__main__":
    run()
