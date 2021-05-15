# warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import in_out

config = in_out.get_parameters()

print(config)


def run(HISTORY_PATH=config["paths"]["history"], FIGURE_DIR=config["dirs"]["figure"]):
    history_df = pd.read_csv(HISTORY_PATH, low_memory=False)
    print(history_df.shape)
    # history_df = history_df[100:]
    # encoder = VenusEncoder()
    #
    AVOID_ZERO = 0.00000001

    f1_scaler = MinMaxScaler(feature_range=(0, 1))
    f1_scaler.fit([[np.log(AVOID_ZERO)], [np.log(1)]])

    history_df["f1_mean"] = np.exp(
        ([history_df["f1_mean"]])[0]
    )

    history_df["f1_max"] = np.exp(
        ([history_df["f1_max"]])[0]
    )
    history_df["f1_min"] = np.exp(
        ([history_df["f1_min"]])[0]
    )

    # history_df["f3_mean"] = 1 / history_df["f3_mean"]
    # history_df["f3_max"] = 1 / history_df["f3_max"]
    # history_df["f3_min"] = 1 / history_df["f3_min"]

    # history_df["g1_6_min"] = history_df["g1_6_min"] * 13560
    # history_df["g1_6_mean"] = history_df["g1_6_mean"] * 13560
    # history_df["g1_6_max"] = history_df["g1_6_max"] * 13560
    #
    # history_df["g1_7_min"] = history_df["g1_7_min"] * 255
    # history_df["g1_7_mean"] = history_df["g1_7_mean"] * 255
    # history_df["g1_7_max"] = history_df["g1_7_max"] * 255
    font = {"size": 16}
    plt.rc("font", **font)

    constraints_min_col = [c for c in history_df.columns if c.startswith("g1_") and c.endswith("min")]
    constraints_min_col = [c.replace("_min", "") for c in constraints_min_col]
    constraints_min_col = [c for c in constraints_min_col if history_df[f"{c}_min"].iloc[-1]]


    objectives = [
        "f1",
        "f2",
    ]
    objectives = objectives + constraints_min_col

    scales = ["linear" for o in objectives]
    y_labels = [
        "Prediction",
        "L2 Perturbation"]
    y_labels = y_labels + constraints_min_col

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
