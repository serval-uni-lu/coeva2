import logging
from tensorflow.keras.models import load_model
import numpy as np
from attacks.coeva2.classifier import Classifier
import seaborn as sns

LOGGER = logging.getLogger()
from utils import Pickler, in_out, load_keras_model

CONFIG = in_out.get_parameters()
import matplotlib.pyplot as plt


def run():
    classifier = Classifier(load_model(in_out.get_parameters()["paths"]["model"]))
    # nb_features = 756
    # nb_samples = 1000000
    # random_samples = np.random.rand((nb_samples * nb_features)).reshape(
    #     nb_samples, nb_features
    # )
    X_initial_states = np.load(CONFIG["paths"]["x_candidates"])
    print(X_initial_states.shape)

    y_pred = classifier.predict_proba(X_initial_states)
    print(y_pred)
    # print(y_pred[:10])
    sns.distplot(y_pred, hist=True, kde=True, kde_kws={"linewidth": 3}, label="Class")
    plt.show()
    print((y_pred == 0).sum())



if __name__ == "__main__":
    run()
