import numpy as np


def gen_mean(history):
    out = {}
    for key in history:
        if key == "g1":
            out[key] = np.array([x.mean() for x in history[key]])
            for i in range(history[key][0].shape[1]):
                out[f"{key}_{i+1}"] = np.array([x[:, i].mean() for x in history[key]])
        else:
            out[key] = np.array([x.mean() for x in history[key]])
    return out


def get_history(results):
    histories = [x.history for x in results]
    gen_means = [gen_mean(x) for x in histories]
    out = {}
    for key in gen_means[0]:
        key_values = np.array([x[key] for x in gen_means])
        out["{}_min".format(key)] = key_values.min(axis=0)
        out["{}_mean".format(key)] = key_values.mean(axis=0)
        out["{}_max".format(key)] = key_values.max(axis=0)
    return out
