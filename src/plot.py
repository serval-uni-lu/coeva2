import pandas as pd
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

df = pd.read_csv("./out/attacks/lcld/for_plot.csv")


# for label, df in df.groupby('method'):
#     df.eps.plot(ax=ax, label=label)
# plt.legend()
# bp = df.groupby('method').plot(kind='kde', ax=ax)

for objective in ["o6", "o7"]:
    fig, ax = plt.subplots(figsize=(16, 12))
    for method in df["method"].unique():
        local = df[df["method"] == method].sort_values("eps")
        local.plot(x='eps', y=objective, ax=ax, label=method, marker="^", markersize=10, linewidth=2)
    ax.set_xscale('log')
    plt.ylabel('success rate')
    plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))
# for method in df["method"].unique():
#     local = df[df["method"] == method].sort_values("eps")
#     local.plot(x='eps', y='o6', ax=ax, label=method, marker="^")
# ax.set_xscale('log')
# plt.show()
