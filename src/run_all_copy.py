import os
import logging
from src.config_parser.config_parser import get_config, merge_parameters

TABULATOR = ">>>"
launch_counter = 0


def launch_script(script):
    global launch_counter
    launch_counter += 1
    logger.info(script)


def run():
    config_dir = config["config_dir"]
    for seed in ["seeds"]:
        logger.info(f"{TABULATOR*1} Running seed {seed} ...")
        for project in config["projects"]:
            logger.info(f"{TABULATOR*2} Running project {project} ...")
            for budget in config["budgets"]:
                logger.info(f"{TABULATOR * 3} Running budget {budget} ...")

                # Run MoEvA
                launch_script(
                    f"python -m src.experiments.united.04_moeva "
                    f"-c {config_dir}/moeva.yaml "
                    f"-c {config_dir}/{project}.yaml "
                    f"-p seed={seed} "
                    f"-p budget={budget} "
                )

                # Run the rest
                for eps in config["eps_list"]:
                    logger.info(f"{TABULATOR * 4} Running eps {eps} ...")

                    for loss_evaluation in config["loss_evaluations"]:
                        logger.info(
                            f"{TABULATOR * 5} Running loss_evaluation {loss_evaluation} ..."
                        )
                        launch_script(
                            f"python -m src.experiments.united.01_pgd_united "
                            f"-c {config_dir}/pgd.yaml "
                            f"-c {config_dir}/{project}.yaml "
                            f"-p seed={seed} "
                            f"-p budget={budget} "
                            f"-p eps={eps} "
                            f"-p loss_evaluation={loss_evaluation} "
                        )


if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    run()
    logger.info(f"{launch_counter} run executed.")
