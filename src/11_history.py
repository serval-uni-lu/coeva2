import warnings
from attacks.history import get_history

warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import pandas as pd

from utils import Pickler, in_out

config = in_out.get_parameters()


def run(
        ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
        HISTORY_PATH=config["paths"]["history"],
):
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)

    results = Pickler.load_from_file(ATTACK_RESULTS_PATH)
    history = get_history(results)
    history_df = pd.DataFrame.from_dict(history)
    history_df.to_csv(HISTORY_PATH)


if __name__ == "__main__":
    run()
