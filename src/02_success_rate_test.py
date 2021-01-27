import warnings

from attacks.coeva2.classifier import Classifier
from attacks.coeva2.feature_encoder import FeatureEncoder
from attacks.coeva2.lcld_constraints import LcldConstraints
from attacks.coeva2.objective_calculator import ObjectiveCalculator

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
from joblib import load
import pandas as pd
from copy import deepcopy
from attacks.objectives import calculate_success_rates
from attacks.venus_encoder import VenusEncoder
from utils import Pickler, in_out

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    OBJECTIVES_PATH=config["paths"]["objectives"],
    THRESHOLD=config["threshold"],
):
    Path(OBJECTIVES_PATH).parent.mkdir(parents=True, exist_ok=True)

    efficient_results = Pickler.load_from_file(ATTACK_RESULTS_PATH)

    classifier = Classifier(load(MODEL_PATH))

    constraints = LcldConstraints(amount_feature_index=config["amount_feature_index"])
    constraints.provision_constraints_min_max(config["paths"]["constraints"])
    constraints.provision_feature_constraints(config["paths"]["features"])

    xl, xu = constraints.get_feature_min_max()
    encoder = FeatureEncoder(
        constraints.get_mutable_mask(),
        constraints.get_feature_type(),
        xl,
        xu,
    )

    objectiveCalculator = ObjectiveCalculator(
        classifier,
        constraints,
        encoder,
        THRESHOLD,
        config["high_amount"],
        config["amount_feature_index"]
    )

    success_rates = objectiveCalculator.success_rate(efficient_results)

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(success_rates.reshape([1, -1]), columns=columns,)
    success_rate_df.to_csv(OBJECTIVES_PATH, index=False)


if __name__ == "__main__":
    run()
