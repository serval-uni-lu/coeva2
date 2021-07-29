import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.experiments.united.utils import get_constraints_from_str
from src.utils import Pickler, in_out
from src.utils.in_out import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    efficient_results = Pickler.load_from_file(config["paths"]["attack_results"])

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    classifier = Classifier(load_model(config["paths"]["model"]))

    scaler = joblib.load(config["paths"]["min_max_scaler"])
    objective_calc = ObjectiveCalculator(
        classifier,
        constraints,
        minimize_class=1,
        thresholds=config["thresholds"],
        min_max_scaler=scaler,
        ml_scaler=scaler,
    )

    x_retrain = objective_calc.get_successful_attacks_results(
        efficient_results, "misclassification", max_inputs=1
    )
    y_retrain = 1 - (
        classifier.predict_proba(scaler.transform(x_retrain))[:, 1]
        >= config["thresholds"]["f1"]
    ).astype(np.float64)
    print(f"Number of results {len(efficient_results)}.")
    print(f"Number of retrain input {x_retrain.shape[0]}.")
    np.save(config["paths"]["x_retrain"], x_retrain)
    np.save(config["paths"]["y_retrain"], y_retrain)


if __name__ == "__main__":
    run()
