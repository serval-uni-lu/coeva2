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
    for seed in config["common"]["seeds"]:
        logger.info(f"{TABULATOR*1} Running seed {seed} ...")
        for project in config["projects"]:
            logger.info(f"{TABULATOR*2} Running project {project['name']} ...")

            # Run MoEvA once
            for f2 in config["common"]["thresholds"]["f2s"]:
                logger.info(f"{TABULATOR*3} Running distance {f2} ...")

                # Run attacks
                for attack in config["common"]["attacks"]:
                    logger.info(f"{TABULATOR * 4} Running attack {attack['name']} ...")
                    attack_param = {"attack": attack}
                    launch_script(
                        f"python -m src.united.{attack['name']}"
                        "-c ./config/00_common"
                        f"-p seed={seed}"
                        f"-p project={project['name']}"
                        f"-p thresholds.f2={f2}"
                        f"-j {attack_param}"
                    )
                    logger.info(
                        f"{TABULATOR * 4} Running success_rate {attack['name']} ..."
                    )
                    if attack["objective_script"] is not None:
                        launch_script(
                            f"python -m src.united.{attack['objective_script']}"
                            "-c ./config/00_common"
                            f"-p seed={seed}"
                            f"-p project={project['name']}"
                            f"-p thresholds.f2={f2}"
                            f"-j {attack_param}"
                        )


if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    run()
    logger.info(f"{launch_counter} run executed.")

