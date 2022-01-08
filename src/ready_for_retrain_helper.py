import joblib
import numpy as np

from src.attacks.moeva2.classifier import ScalerClassifier
from src.attacks.objective_calculator import ObjectiveCalculator
from src.config_parser.config_parser import get_config
from src.experiments.united.utils import get_constraints_from_str


def run():
    project = config.get("project")
    model_name = config.get("model_name")
    model_path = f"./models/{project}/{model_name}.model"
    scaler_path = f"./models/{project}/{model_name}_scaler.joblib"
    classifier = ScalerClassifier(model_path, scaler_path)
    constraints = get_constraints_from_str(project)()
    threshold = config.get("classification_threshold")
    scaler = joblib.load(scaler_path)
    objective_calculator = ObjectiveCalculator(
        classifier,
        constraints,
        thresholds={
            "model": threshold if threshold is not None else 0.5,
            "distance": config.get("distance"),
        },
        fun_distance_preprocess=scaler.transform,
        norm=config.get("norm"),
    )
    x_clean = np.load(f"./data/{project}/{model_name}_X_{config.get('candidates')}_candidate")
    y_clean = np.load(f"./data/{project}/{model_name}_y_{config.get('candidates')}_candidate")
    for x_attack in config.get("x_attacks", []):
        x_attacks = np.load(x_attack["path"])
        x_adv, x_adv_i = objective_calculator.get_successful_attacks(
            x_clean,
            y_clean,
            x_attacks,
            preferred_metrics="misclassification",
            order="asc",
            max_inputs=1,
            return_index_success=True,
        )
        x_adv_i_r = [i for i in np.arange(x_clean.shape[0]) if i not in x_adv_i]
        x_adv = np.array(x_adv)
        x_adv[x_adv_i_r] = x_clean[x_adv_i_r][np.newaxis, :, :]
        x_adv = np.concatenate(x_adv)
        print(x_adv.shape)
        np.save(x_attack, x_adv)


if __name__ == "__main__":
    config = get_config()

    run()
