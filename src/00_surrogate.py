from joblib import load
import numpy as np
from sklearn.model_selection import train_test_split
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
    MODEL_PATH=config["paths"]["model"],
    SURROGATE_PATH=config["paths"]["surrogate"],
    SCALER_PATH=config["paths"]["scaler"],
    THRESHOLD=config["threshold"],
    TRAIN_TEST_DATA_DIR=config["dirs"]["train_test_data"],
    X_ATTACK_CANDIDATES_PATH=config["paths"]["x_candidates"],
    X_SURROGATE_CANDIDATES_PATH=config["paths"]["x_surrogate_candidates"],
):
    # ----- Load

    X_train = np.load("{}/X_train.npy".format(TRAIN_TEST_DATA_DIR))
    original_model = load(MODEL_PATH)
    scaler = Pickler.load_from_file(SCALER_PATH)

    # ----- Label From Previous Model

    y_train = (original_model.predict_proba(X_train)[:, 1] >= THRESHOLD).astype(bool)

    # ----- Split and Scale

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # ----- Model Definition

    model = Sequential()
    model.add(Dense(X_train.shape[1], activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(56, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(28, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=["accuracy", tf.metrics.AUC()])

    # ----- Model Training

    r = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=2,
        batch_size=16,
    )

    # ----- Print Test

    y_test_pred = model.predict(X_test)
    y_pred = y_test_pred.round()
    print_score(y_test, y_pred)

    # ----- Save Model

    tf.keras.models.save_model(
        model,
        SURROGATE_PATH,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    # ----- CANDIDATES

    X_candidates = np.load(X_ATTACK_CANDIDATES_PATH)
    print("{} Candidates".format(X_candidates.shape[0]))
    y_candidates = model.predict(X_candidates).round()
    index = y_candidates == 1
    X_candidates = X_candidates[index[:, 0]]
    print("{} Candidates".format(X_candidates.shape[0]))
    X_candidates.save(X_SURROGATE_CANDIDATES_PATH)


if __name__ == "__main__":
    run()
