import numpy as np
import pandas as pd

np.random.seed(205)
import tensorflow as tf

tf.random.set_seed(206)
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping

x_train = np.load("./data/lcld/X_train.npy")
x_test = np.load("./data/lcld/X_test.npy")
y_train = np.load("./data/lcld/y_train.npy")
y_test = np.load("./data/lcld/y_test.npy")

scaler = MinMaxScaler()
scaler.fit(np.concatenate((x_train, x_test)))

x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)
features = pd.read_csv("./data/lcld/features.csv")


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


def create_model():

    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
    )
    return model


def train_model(x_train_s, y_train):
    x_train_local, x_val_local, y_train_local, y_val_local = train_test_split(
        x_train_s,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )
    model = create_model()
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
    model.fit(
        x=x_train_local,
        y=y_train_local,
        epochs=100,
        batch_size=512,
        validation_data=(x_val_local, y_val_local),
        verbose=1,
        callbacks=[early_stop],
    )
    return model


strong = train_model(x_train_s, y_train)
weak = train_model(x_train_s[:, features["augmentation"]], y_train)

for i, model in enumerate([strong, weak]):
    if i == 0:
        y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
    else:
        y_proba = model.predict_proba(scaler.transform(x_test)[:, features["augmentation"]]).reshape(-1)
    mccs = [matthews_corrcoef(y_test, (y_proba >= t / 100).astype(int)) for t in range(100)]
    threshold = np.argmax(mccs) / 100
    print(f"{threshold}")
    y_pred = (y_proba >= threshold).astype(int)
    print_score(y_test, y_pred)


#
# candidates_index = (y_test == 1) * (y_test == y_pred)
# print(candidates_index.shape)
# print(candidates_index.sum())
# X_candidate = x_test[candidates_index, :]
# print(X_candidate.shape)
#
# np.save("./data/lcld/x_attack_candidates.npy", X_candidate)
#
# tf.keras.models.save_model(
#     model,
#     "./models/lcld/nn.model",
#     overwrite=True,
#     include_optimizer=True,
#     save_format=None,
#     signatures=None,
#     options=None,
# )
# joblib.dump(scaler, "./models/lcld/scaler.joblib")
