import numpy as np


def gen_mean(history):
    out = {}
    for key in history:
        out[key] = np.array([x.mean() for x in history[key]])
    return out


def get_history(results):
    histories = [x.history for x in results]
    gen_means = [gen_mean(x) for x in histories]
    out = {}
    for key in histories[1]:
        key_values = np.array([x[key] for x in gen_means])
        out["{}_min".format(key)] = key_values.min(axis=0)
        out["{}_mean".format(key)] = key_values.mean(axis=0)
        out["{}_max".format(key)] = key_values.max(axis=0)
    return out
