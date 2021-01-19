import pandas as pd
from utils import in_out
import matplotlib.pyplot as plt
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


def plot_single(ax, data, method):
    x = [i + 1 for i in range(n_objectives)]
    y = [data[objectives_col[i]] for i in range(n_objectives)]

    a,  = plt.plot(
        x, y, method["style"], axes=ax,
    )
    a.set_label(method["name"])


def plot_multiple(ax, data):
    data.boxplot(ax=ax, column=objectives_col)


def run(METHODS=config["methods"], OUTPUT_PATH=config["output_path"]):

    fig, ax = plt.subplots()
    for method in METHODS:
        data = pd.read_csv(method["path"])
        if len(data) > 1:
            plot_multiple(ax, data)
        else:
            plot_single(ax, data, method)

    # ax.legend(["Multi-objective (NSGA-II)","Multi-objective (NSGA-II)"])
    ax.legend()
    plt.ylabel("Success rate")
    plt.xlabel("Objectives")
    fig.set_size_inches(11, 8)
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.13, right=0.92, wspace = 0.3, hspace=0.3)

    # major_ticks = np.arange(0.3, 0.6, 0.05)
    # minor_ticks = np.arange(0.3, 0.6, 0.01)

    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    # ax.grid(which='minor', color='black', alpha=0.2)
    # ax.grid(which='major', color='black', alpha=0.4)
    plt.savefig(OUTPUT_PATH)


if __name__ == "__main__":
    run()
