import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from utils import Pickler, Datafilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from attacks.coeva2.lcld_constraints import LcldConstraints
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from utils import in_out

config = in_out.get_parameters()


def print_score(label, prediction):
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end="")
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
    tf.compat.v1.disable_eager_execution()

    X_train = np.load("{}/X_train.npy".format(TRAIN_TEST_DATA_DIR))
    y_train = np.load("{}/y_train.npy".format(TRAIN_TEST_DATA_DIR))
    X_test = np.load("{}/X_test.npy".format(TRAIN_TEST_DATA_DIR))
    y_test = np.load("{}/y_test.npy".format(TRAIN_TEST_DATA_DIR))
    scaler = joblib.load(SCALER_PATH)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # ----- Split and Scale

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train
    )

    X_train = np.array(X_train).astype(np.float32)
    X_val = np.array(X_val).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.array(y_train).astype(np.float32))
    y_val = tf.keras.utils.to_categorical(np.array(y_val).astype(np.float32))

    # ----- Model Definition

    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=X_train.shape[1]))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.001))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    sgd = Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=sgd)

    # model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #               loss='binary_crossentropy',
    #               metrics=["accuracy", AUC()])

    # ----- Model Training

    r = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
    )

    # ----- Print Test

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)
    print_score(y_test, y_pred)

    # ----- Save Model

    tf.keras.models.save_model(
        model,
        SURROGATE_PATH,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )

    X_test = scaler.fit_transform(np.load(config["paths"]["x_candidates"]))
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)
    y_test = np.ones(X_test.shape[0])

    X_test, y_test, y_pred = Datafilter.filter_correct_prediction(
        X_test, y_test, y_pred
    )
    X_test, y_test, y_pred = Datafilter.filter_by_target_class(
        X_test, y_test, y_pred, 1
    )
    X_test = X_test[np.random.permutation(X_test.shape[0])]

    # Removing x that violates constraints
    constraints_evaluator = LcldConstraints(
        # config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    constraints = constraints_evaluator.evaluate(scaler.inverse_transform(X_test))
    # Scaling tolerance = 5
    constraints[:, 0] = constraints[:, 0] - 1
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_test = X_test[(1 - constraints_violated).astype(bool)]
    print("{} candidates.".format(X_test.shape[0]))
    np.save(X_SURROGATE_CANDIDATES_PATH, X_test)


if __name__ == "__main__":
    run()
