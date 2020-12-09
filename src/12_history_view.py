import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attacks.venus_encoder import VenusEncoder
from utils import Pickler, in_out

config = in_out.get_parameters()


def run(
    HISTORY_PATH=config["paths"]["history"],
):

    history_df = pd.read_csv(HISTORY_PATH, low_memory=False)

    # history_df = history_df[100:]
    encoder = VenusEncoder()

    history_df["f1_mean"] = np.exp(encoder.f1_scaler.inverse_transform([history_df["f1_mean"]])[0])
    history_df["f1_max"] = np.exp(encoder.f1_scaler.inverse_transform([history_df["f1_max"]])[0])
    history_df["f1_min"] = np.exp(encoder.f1_scaler.inverse_transform([history_df["f1_min"]])[0])

    history_df["f3_mean"] = 1/history_df["f3_mean"]
    history_df["f3_max"] = 1/history_df["f3_max"]
    history_df["f3_min"] = 1/history_df["f3_min"]

    for key in ["f1", "f2", "f3", "g1"]:
        fig, ax = plt.subplots()
        ax.plot(history_df.index, history_df["{}_mean".format(key)], '-', color="white")
        ax.fill_between(x=history_df.index, y1=history_df["{}_min".format(key)], y2=history_df["{}_max".format(key)])
        plt.show()


if __name__ == "__main__":
    run()
