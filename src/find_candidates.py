import joblib
import numpy as np

from src.utils import Pickler, in_out, filter_initial_states

from sklearn.metrics import confusion_matrix


def run():

    threshold = 0.5
    model = joblib.load("./models/malware/model_small.joblib")
    x_test = np.load("./data/malware/training_data_small/X_test.npy")
    y_test = np.load("./data/malware/training_data_small/y_test.npy")

    cm = confusion_matrix(y_test, model.predict_proba(x_test)[:, 1] >= threshold)
    print(cm)
    x_test_1 = x_test[y_test == 1]
    y_test_1 = y_test[y_test == 1]

    y_proba_1 = model.predict_proba(x_test_1)
    y_pred_1 = (y_proba_1[:, 1] >= threshold).astype(np.int)

    x_candidates = x_test_1[y_pred_1 == 1]

    print(f"There is {x_candidates.shape[0]} candidates.")

    np.save("./data/malware/x_attack_candidate_small.npy", x_candidates)


if __name__ == "__main__":
    run()
