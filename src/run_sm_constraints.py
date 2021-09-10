import json
import os
import logging
import subprocess

import numpy as np

from src.config_parser.config_parser import get_config, merge_parameters

TABULATOR = ">>>"
launch_counter = 0


def launch_script(script):
    global launch_counter
    launch_counter += 1
    logger.info(script)
    subprocess.run(script)


def run():
    config_dir = config["config_dir"]
    for seed in config["seeds"]:
        logger.info(f"{TABULATOR*1} Running seed {seed} ...")
        for project in config["projects"]:
            logger.info(f"{TABULATOR*2} Running project {project} ...")
            for budget in config["budgets"]:
                logger.info(f"{TABULATOR * 3} Running budget {budget} ...")
                for i_augmentation in np.arange(2, config["max_augmentation"]):
                    logger.info(f"{TABULATOR * 4} Running augmentation {i_augmentation} ...")
                    path_prefix = config.get("path_prefix")
                    paths = {
                        "paths": {
                            "model": f"./models/{path_prefix}/sm_constraints/nn_augmented_{i_augmentation}.model",
                            "features": f"./data/{path_prefix}/sm_constraints/features_augmented_{i_augmentation}.csv",
                            "constraints": f"./data/{path_prefix}/sm_constraints/constraints_augmented_{i_augmentation}.csv",
                            "x_candidates": f"./data/{path_prefix}/sm_constraints/x_candidates_augmented_{i_augmentation}.npy",
                            "min_max_scaler": f"./models/{path_prefix}/sm_constraints/scaler_augmented_{i_augmentation}.joblib",
                            "ml_scaler": f"./models/{path_prefix}/sm_constraints/scaler_augmented_{i_augmentation}.joblib",
                            "important_features": f"./data/{path_prefix}/sm_constraints/important_features_{i_augmentation}.npy"
                        }
                    }
                    paths_str = json.dumps(paths, separators=(',', ':'))

                    if "moeva" in config["attacks"]:
                        logger.info(f"{TABULATOR * 5} Running MoEvA ...")
                        eps_list = {"eps_list": config['eps_list']}
                        eps_list_str = json.dumps(eps_list, separators=(',', ':'))
                        launch_script([
                            "python", "-m", "src.experiments.united.04_moeva",
                            "-c", f"{config_dir}/moeva.yaml",
                            "-c", f"{config_dir}/{project}.yaml",
                            "-j", paths_str,
                            "-p", f"seed={seed}",
                            "-p", f"budget={budget}",
                            "-j", eps_list_str,
                            ]
                        )

                    # Run the rest
                    if "pgd" in config["attacks"]:
                        logger.info(f"{TABULATOR * 5} Running pgd ...")
                        for eps in config["eps_list"]:
                            logger.info(f"{TABULATOR * 6} Running eps {eps} ...")

                            for loss_evaluation in config["loss_evaluations"]:
                                logger.info(
                                    f"{TABULATOR * 7} Running loss_evaluation {loss_evaluation} ..."
                                )
                                launch_script([
                                    "python", f"-m", f"src.experiments.united.01_pgd_united",
                                    "-c", f"{config_dir}/pgd.yaml",
                                    "-c", f"{config_dir}/{project}.yaml",
                                    "-j", paths_str,
                                    "-p", f"seed={seed}",
                                    "-p", f"budget={budget}",
                                    "-p", f"eps={eps}",
                                    "-p", f"loss_evaluation={loss_evaluation}"]
                                )


if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    run()
    logger.info(f"{launch_counter} run executed.")
