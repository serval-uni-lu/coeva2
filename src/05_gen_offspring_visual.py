import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from utils import in_out

monochrome = (
    cycler("color", ["k"])
    * cycler("marker", ["o", "v", "s", "*"])
    * cycler("markersize", [22])
)
plt.rc("axes", prop_cycle=monochrome)
font = {"family": "normal", "size": 22}

plt.rc("font", **font)

config = in_out.get_parameters()


N_OBJECTIVES = 4


def run(OBJECTIVES_PATH=config["paths"]["objectives"]):
    df = pd.read_csv(OBJECTIVES_PATH)
    df.sort_values(by=["n_offsprings"], inplace=True)
    fig, ax = plt.subplots()
    for i in range(N_OBJECTIVES):
        df.plot(ax=ax, x="n_offsprings", y="o{}".format(i + 1))

    plt.ylabel("Success rate")
    plt.show()


if __name__ == "__main__":
    run()
