import warnings

import joblib
import pandas as pd

from attacks.moeva2.classifier import Classifier
from attacks.moeva2.objective_calculator import ObjectiveCalculator
from examples.botnet.botnet_constraints import BotnetConstraints
from pathlib import Path

from utils import Pickler, in_out

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    efficient_results = Pickler.load_from_file(config["paths"]["attack_results"])

    constraints = BotnetConstraints(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    classifier = Classifier(joblib.load(config["paths"]["model"]))

    objective_calc = ObjectiveCalculator(
        classifier, constraints, minimize_class=1, thresholds=config["thresholds"]
    )
    success_rates = objective_calc.success_rate_genetic(efficient_results)

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(
        success_rates.reshape([1, -1]),
        columns=columns,
    )
    success_rate_df.to_csv(config["paths"]["objectives"], index=False)


if __name__ == "__main__":
    run()
