import logging
from tensorflow.keras.models import load_model
import numpy as np
from attacks.coeva2.classifier import Classifier
import seaborn as sns

LOGGER = logging.getLogger()
from utils import Pickler, in_out, load_keras_model
from sklearn.metrics import confusion_matrix
import pickle

CONFIG = in_out.get_parameters()
import matplotlib.pyplot as plt
from utils import Datafilter


def run():
    classifier = Classifier(load_model(in_out.get_parameters()["paths"]["model"]))
    nb_features = 756
    nb_samples = 10000
    random_samples = np.random.rand((nb_samples * nb_features)).reshape(
        nb_samples, nb_features
    )
    with open('../models/botnet/scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    X_test = (np.load("../data/botnet/x_test.npy"))
    X_test_scaled = scaler.transform(X_test)
    y_test = np.load("../data/botnet/y_test.npy")

    y_pred_proba = classifier.predict_proba(X_test_scaled).reshape(-1)

    y_pred = (y_pred_proba >= CONFIG["threshold"]).astype(bool)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ----- SAVE X correctly rejected loans and respecting constraints

    X_test, y_test, y_pred = Datafilter.filter_correct_prediction(
        X_test, y_test, y_pred
    )
    X_test, y_test, y_pred = Datafilter.filter_by_target_class(
        X_test, y_test, y_pred, 1
    )
    X_test = X_test[np.random.permutation(X_test.shape[0])]
    print(X_test.shape)

    y_pred_proba = classifier.predict_proba(scaler.transform(X_test)).reshape(-1)
    print(y_pred_proba.min())

    # X_test = np.load(CONFIG["paths"]["x_candidates"] )

    X_test[:,597] = X_test[:,597] + 300
    y_pred_proba = classifier.predict_proba(scaler.transform(X_test)).reshape(-1)
    print(y_pred_proba.min())

    y_pred_proba = classifier.predict_proba(scaler.transform(random_samples)).reshape(-1)
    print(y_pred_proba.min())


    # X_initial_states = np.load(CONFIG["paths"]["x_candidates"])
    # print(X_initial_states.shape)

    # y_pred = classifier.predict_proba(X_initial_states)
    # print(y_pred)
    # # print(y_pred[:10])
    sns.distplot(y_pred_proba, hist=True, kde=True, kde_kws={"linewidth": 3}, label="Class")
    plt.show()
    # print((y_pred == 0).sum())



if __name__ == "__main__":
    run()
