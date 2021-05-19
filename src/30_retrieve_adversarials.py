import warnings

from attacks.coeva2.classifier import Classifier
from src.examples.botnet.botnet_constraints import LcldConstraints
from attacks.coeva2.objective_calculator import ObjectiveCalculator

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
from joblib import load
from utils import Pickler, in_out
import numpy as np

config = in_out.get_parameters()
import pymoo

print(pymoo.__version__)


def run(
    MODEL_PATH=config["paths"]["model"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    OBJECTIVES_PATH=config["paths"]["objectives"],
    THRESHOLD=config["threshold"],
):
    Path(OBJECTIVES_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Load and create object
    efficient_results = Pickler.load_from_file(ATTACK_RESULTS_PATH)
    classifier = Classifier(load(MODEL_PATH))
    constraints = LcldConstraints(
        # config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    objective_calculator = ObjectiveCalculator(
        classifier,
        constraints,
        THRESHOLD,
        # config["high_amount"],
        # config["amount_feature_index"]
    )
    adv = objective_calculator.get_successful_attacks(efficient_results)

    np.save(config["paths"]["x_adv"], adv)


if __name__ == "__main__":
    run()
