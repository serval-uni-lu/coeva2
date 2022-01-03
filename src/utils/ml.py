import numpy as np


def convert_2d_score(y_score):
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)
    if y_score.shape[1] == 1:
        y_score = np.concatenate((1 - y_score, y_score), axis=1)

    return y_score
