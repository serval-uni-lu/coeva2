from src.config_parser.config_parser import get_config
import logging
import time


def run():
    logger.info("Simulating run.")
    print(config)
    time.sleep(1)
    logger.info("Run finished.")


if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    run()
