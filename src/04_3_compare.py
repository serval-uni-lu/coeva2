import pandas as pd
from utils import in_out
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from cycler import cycler
import numpy as np

config = in_out.get_parameters()

# ----- CONSTANT
n_objectives = 3
objectives_col = ["o{}".format(i + 2) for i in range(n_objectives)]

monochrome = cycler("color", ["k"]) * cycler("markersize", [11])
plt.rc("axes", prop_cycle=monochrome)
font = {"size": 16}

plt.rc("font", **font)


def plot_single(gs, data, method):
    ax0 = plt.subplot(gs[0])
    x = [1]
    y = [data["o1"]]
    (a,) = plt.plot(
        x,
        y,
        method["style"],
        axes=ax0,
    )
    # a.set_label(method["name"])

    ax1 = plt.subplot(gs[1])
    x = [i + 1 for i in range(n_objectives)]
    y = [data[objectives_col[i]] for i in range(n_objectives)]

    (a,) = plt.plot(
        x,
        y,
        method["style"],
        axes=ax1,
    )
    a.set_label(method["name"])


def plot_multiple(gs, data):
    ax0 = plt.subplot(gs[0])
    data.boxplot(ax=ax0, column=["o1"])
    ax1 = plt.subplot(gs[1])
    data.boxplot(ax=ax1, column=objectives_col)


def run(METHODS=config["methods"], OUTPUT_PATH=config["output_path"]):
    fig = plt.figure(constrained_layout=True)
    gs = grd.GridSpec(1, 2, figure=fig, width_ratios=[0.25, 0.75])
    axs = [fig.add_subplot(gs[0, i]) for i in [0, 1]]
    for method in METHODS:
        data = pd.read_csv(method["path"])
        if len(data) > 1:
            plot_multiple(gs, data)
        else:
            plot_single(gs, data, method)

    plt.legend(loc=3)
    plt.subplot(gs[0])
    plt.ylabel("Success rate")
    fig.text(0.5, 0.04, "Objectives", ha="center")
    fig.set_size_inches(11, 8)
    # plt.subplots_adjust(
    #     top=0.95, bottom=0.12, left=0.13, right=0.92, wspace=0.3, hspace=0.3
    # )

    plt.savefig(OUTPUT_PATH)


if __name__ == "__main__":
    run()
