import pandas as pd

import pandas as pd
import numpy as np
import glob

pd.set_option('display.max_columns', 15)

path = r'../out/attacks/07_coeva2_all'
all_files = glob.glob(path + "/*.csv")

dfs = []

for filename in all_files:
    df = pd.read_csv(filename)
    dfs.append(df)

history = {

    "f1_min": np.array([df["f1_min"].to_numpy() for df in dfs]).min(axis=0),
    "f1_mean": np.array([df["f1_mean"].to_numpy() for df in dfs]).mean(axis=0),
    "f1_max": np.array([df["f1_max"].to_numpy() for df in dfs]).max(axis=0),
    "f2_min": np.array([df["f2_min"].to_numpy() for df in dfs]).min(axis=0),
    "f2_mean": np.array([df["f2_mean"].to_numpy() for df in dfs]).mean(axis=0),
    "f2_max": np.array([df["f2_max"].to_numpy() for df in dfs]).max(axis=0),
    "f3_min": np.array([df["f3_min"].to_numpy() for df in dfs]).min(axis=0),
    "f3_mean": np.array([df["f3_mean"].to_numpy() for df in dfs]).mean(axis=0),
    "f3_max": np.array([df["f3_max"].to_numpy() for df in dfs]).max(axis=0),
    "g1_min": np.array([df["g1_min"].to_numpy() for df in dfs]).min(axis=0),
    "g1_mean": np.array([df["g1_mean"].to_numpy() for df in dfs]).mean(axis=0),
    "g1_max": np.array([df["g1_max"].to_numpy() for df in dfs]).max(axis=0),
}

history_df = pd.DataFrame.from_dict(history)
history_df.to_csv("../out/attacks/07_coeva2_all/history_all.csv")
print(history_df)
