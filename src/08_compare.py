import pandas as pd
from utils import in_out
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

config = in_out.get_parameters()

# ----- CONSTANT
n_objectives = 3
objectives_col = ["o{}".format(i + 2) for i in range(n_objectives)]

monochrome = cycler("color", ["k"]) * cycler("markersize", [16])
plt.rc("axes", prop_cycle=monochrome)
font = {"family": "normal", "size": 22}

plt.rc("font", **font)


def plot_single(ax, data, style):
    for i in range(n_objectives):
        plt.plot(
            i + 1, data[objectives_col[i]], style, axes=ax,
        )


def plot_multiple(ax, data):
    data.boxplot(ax=ax, column=objectives_col)


def run(METHODS=config["methods"]):
    fig, ax = plt.subplots()
    for method in METHODS:
        data = pd.read_csv(method["path"])
        if len(data) > 1:
            plot_multiple(ax, data)
        else:
            plot_single(ax, data, method["style"])

    ax.legend(["Original weights"])
    plt.ylabel("Success rate")

    # major_ticks = np.arange(0.3, 0.6, 0.05)
    # minor_ticks = np.arange(0.3, 0.6, 0.01)

    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    # ax.grid(which='minor', color='black', alpha=0.2)
    # ax.grid(which='major', color='black', alpha=0.4)
    plt.show()


if __name__ == "__main__":
    run()
