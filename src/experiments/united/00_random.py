from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.experiments.united.utils import get_constraints_from_str
from src.utils import in_out, filter_initial_states, sample_in_norm
from src.utils.in_out import load_model

config = in_out.get_parameters()


def random_sample_hyperball(n, d):
    u = np.random.normal(0, 1, (d + 2) * n).reshape(n, d + 2)
    norm = np.linalg.norm(u, axis=1)
    u = u / norm.reshape(-1, 1)
    x = u[:, 0:d]
    return x


def apply_random_perturbation(
    x_init, n_repetition, mask, eps, norm, a_min, a_max, mask_int
):
    x_perturbed = np.repeat(x_init[np.newaxis, :], n_repetition, axis=0)

    x_perturbation = sample_in_norm(n_repetition, mask.sum(), eps, norm)

    x_perturbed[:, mask] = x_perturbed[:, mask] + x_perturbation

    x_perturbed = np.clip(x_perturbed, a_min, a_max)

    return x_perturbed


def run():
    Path(config["paths"]["attack_results"]).parent.mkdir(parents=True, exist_ok=True)

    scaler = joblib.load(config["paths"]["min_max_scaler"])
    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )

    eps = config["thresholds"]["f2"]
    norm = config["norm"]

    x_initial_scaled = scaler.transform(x_initial)

    np.random.seed(config["seed"])

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    mask = constraints.get_mutable_mask()

    iterable = x_initial
    if config["verbose"] > 0:
        iterable = tqdm(iterable, total=len(x_initial_scaled))

    mask_int = constraints.get_feature_type() != "real"

    def apply_one(x_init):
        classifier = Classifier(load_model(config["paths"]["model"]))

        objective_calc = ObjectiveCalculator(
            classifier,
            constraints,
            minimize_class=1,
            thresholds=config["thresholds"],
            min_max_scaler=scaler,
            ml_scaler=scaler,
        )

        x_perturbed = apply_random_perturbation(
            scaler.transform(x_init.reshape(1, -1)).reshape(-1),
            config["n_repetition"],
            mask,
            eps,
            norm,
            0,
            1,
            mask_int,
        )
        x_perturbed = scaler.inverse_transform(x_perturbed)
        x_perturbed[:, mask_int] = np.round(x_perturbed[:, mask_int])

        return objective_calc.at_least_one(
            x_init,
            x_perturbed,
        )

    if config["n_jobs"] == 1:
        at_least_one = np.array([apply_one(x_init) for x_init in iterable])
    else:
        # Parallel run
        at_least_one = np.array(
            Parallel(n_jobs=config["n_jobs"])(
                delayed(apply_one)(x_init) for x_init in iterable
            )
        )

    # print(at_least_one)

    success_rates = at_least_one.mean(axis=0)

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(
        success_rates.reshape([1, -1]),
        columns=columns,
    )
    success_rate_df.to_csv(config["paths"]["objectives"], index=False)
    print(success_rate_df)


if __name__ == "__main__":
    run()
