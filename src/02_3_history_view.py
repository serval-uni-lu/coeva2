import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import in_out

config = in_out.get_parameters()


def run(HISTORY_PATH=config["paths"]["history"], FIGURE_DIR=config["dirs"]["figure"]):

    history_df = pd.read_csv(HISTORY_PATH, low_memory=False)

    # history_df = history_df[100:]
    # encoder = VenusEncoder()
    #
    # history_df["f1_mean"] = np.exp(
    #     encoder.f1_scaler.inverse_transform([history_df["f1_mean"]])[0]
    # )
    # history_df["f1_max"] = np.exp(
    #     encoder.f1_scaler.inverse_transform([history_df["f1_max"]])[0]
    # )
    # history_df["f1_min"] = np.exp(
    #     encoder.f1_scaler.inverse_transform([history_df["f1_min"]])[0]
    # )

    history_df["f3_mean"] = 1 / history_df["f3_mean"]
    history_df["f3_max"] = 1 / history_df["f3_max"]
    history_df["f3_min"] = 1 / history_df["f3_min"]

    font = {"size": 16}
    plt.rc("font", **font)

    objectives = ["f1", "f2", "f3", "g1"]
    scales = ["linear", "linear", "linear", "linear"]
    y_labels = ["Prediction", "L2 Perturbation", "Overdraft", "Constraint violation"]
    for i, key in enumerate(objectives):
        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
        ax = axs
        ax.plot(
            history_df.index,
            history_df["{}_mean".format(key)],
            color="red",
            linewidth=4,
        )
        ax.fill_between(
            x=history_df.index,
            y1=history_df["{}_min".format(key)],
            y2=history_df["{}_max".format(key)],
        )
        ax.set_yscale(scales[i])
        ax.set_xlabel("Generation")
        ax.set_ylabel(y_labels[i])
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}/fig_{key}.pdf")
    


if __name__ == "__main__":
    run()
