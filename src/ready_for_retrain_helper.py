import joblib
import numpy as np

from src.attacks.moeva2.classifier import ScalerClassifier
from src.attacks.objective_calculator import ObjectiveCalculator
from src.config_parser.config_parser import get_config
from src.experiments.united.utils import get_constraints_from_str
import gc
import os

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
        n_jobs=100
    )
    x_clean = np.load(f"./data/{project}/{model_name}_X_{config.get('candidates')}_candidates.npy")
    y_clean = np.load(f"./data/{project}/{model_name}_y_{config.get('candidates')}_candidates.npy")
    outs = []
    for x_attack_i, x_attack in enumerate(config.get("x_attacks", [])):
        print(f"------ DOING{x_attack_i}")
        out_path = f"{x_attack}_small.npy"
        if os.path.exists(out_path):
            out = np.load(out_path)
            print(f"Loaded with shape{out.shape}")
        else:
            x_clean_l = x_clean[x_attack_i*1600:(x_attack_i+1)*1600]
            y_clean_l = y_clean[x_attack_i*1600:(x_attack_i+1)*1600]
            x_attacks = np.load(x_attack)
            x_adv, x_adv_i = objective_calculator.get_successful_attacks(
                x_clean_l,
                y_clean_l,
                x_attacks,
                preferred_metrics="misclassification",
                order="asc",
                max_inputs=1,
                return_index_success=True,
            )
            out = []
            for i in range(x_clean_l.shape[0]):
                if i in x_adv_i:
                    out.append(x_adv[i])
                else:
                    out.append(x_clean_l[i][np.newaxis, :])

            x_attacks = None
            gc.collect()        
            out = np.concatenate(out)
            np.save(f"{x_attack}_small", out)
        outs.append(out)

    x_all = np.concatenate(outs)
    x_all = x_all[:, np.newaxis, :]
    print(x_clean.shape)
    print(x_all.shape)
    np.save(f"/scratch/users/tsimonetto/out_malware/X_all.npy", x_all)

if __name__ == "__main__":
    config = get_config()

    run()
