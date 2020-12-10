import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
from art.classifiers import KerasClassifier
from sklearn.model_selection import train_test_split

from attacks import venus_constraints
from utils import Pickler, Datafilter
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from utils import in_out

config = in_out.get_parameters()

def print_score(label, prediction):
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end='')
    print(f"\tPrecision Score: {precision_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tRecall Score: {recall_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tF1 score: {f1_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tMCC score: {matthews_corrcoef(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")


def run(
    SURROGATE_PATH=config["paths"]["surrogate"],
    SCALER_PATH=config["paths"]["scaler"],
    THRESHOLD=config["threshold"],
    TRAIN_TEST_DATA_DIR=config["dirs"]["train_test_data"],
    X_SURROGATE_CANDIDATES_PATH=config["paths"]["x_surrogate_candidates"],
):
    # ----- Load and Scale

    X_test = np.load("{}/X_test.npy".format(TRAIN_TEST_DATA_DIR))
    y_test = np.load("{}/y_test.npy".format(TRAIN_TEST_DATA_DIR))
    scaler = Pickler.load_from_file(SCALER_PATH)
    X_test = scaler.transform(X_test)
    X_candidates = np.load(X_SURROGATE_CANDIDATES_PATH)


    # ----- Load Model

    model = tf.keras.models.load_model(SURROGATE_PATH)

    # ----- Print Test


    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= THRESHOLD).astype(bool)[:, 0]
    print_score(y_test, y_pred)

    # ----- Attack
    tf.compat.v1.disable_eager_execution()
    attack = ProjectedGradientDescent(estimator=KerasClassifier(model), eps=0.2)
    x_test_adv = attack.generate(x=X_candidates)
    print(x_test_adv.shape)



if __name__ == "__main__":
    run()
