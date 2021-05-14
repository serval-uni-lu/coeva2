import logging
from tensorflow.keras.models import load_model
import numpy as np
from attacks.coeva2.classifier import Classifier
import seaborn as sns

LOGGER = logging.getLogger()
from utils import Pickler, in_out, load_keras_model
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score
import pickle
import joblib

CONFIG = in_out.get_parameters()
import matplotlib.pyplot as plt
from utils import Datafilter


def run():
    classifier = joblib.load("../models/botnet/random_forest.joblib")
    # nb_features = 756
    # nb_samples = 100000
    # random_samples = np.random.rand((nb_samples * nb_features)).reshape(
    #     nb_samples, nb_features
    # )

    X_test = np.load("../data/botnet/x_test.npy")
    y_test = np.load("../data/botnet/y_test.npy")

    y_pred_proba = classifier.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= CONFIG["threshold"]).astype(bool)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    mcc = np.array(
        [
            [
                t / 100,
                matthews_corrcoef(y_test, (y_pred_proba[:, 1] >= t / 100).astype(bool)),
                f1_score(y_test, (y_pred_proba[:, 1] >= t / 100).astype(bool)),
            ]
            for t in range(100)
        ]
    )
    # print(mcc)
    # print(mcc)
    print(f"Max mcc {np.argmax(mcc[:,1])}")
    print(f"Max f1 {np.argmax(mcc[:,2])}")

    # plt.plot(mcc[:, 0], mcc[:, 1])
    # plt.plot(mcc[:, 0], mcc[:, 2])
    # plt.show()

    # ----- SAVE X correctly rejected loans and respecting constraints

    # X_test, y_test, y_pred = Datafilter.filter_correct_prediction(
    #     X_test, y_test, y_pred
    # )
    # X_test, y_test, y_pred = Datafilter.filter_by_target_class(
    #     X_test, y_test, y_pred, 1
    # )
    # X_test = X_test[np.random.permutation(X_test.shape[0])]
    # print(X_test.shape)
    #
    # y_pred_proba = classifier.predict_proba((X_test))[:,1]
    # print(y_pred_proba.min())
    #
    # np.save(CONFIG["paths"]["x_candidates"], X_test)

    # X_initial_states = np.load(CONFIG["paths"]["x_candidates"])
    # print(X_initial_states.shape)

    # y_pred = classifier.predict_proba(X_initial_states)
    # print(y_pred)
    # # print(y_pred[:10])
    # sns.distplot(y_pred, hist=True, kde=True, kde_kws={"linewidth": 3}, label="Class")
    # plt.show()
    # print((y_pred == 0).sum())


if __name__ == "__main__":
    run()
