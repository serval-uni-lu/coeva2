from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

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
features = pd.read_csv("./data/lcld/features.csv")
feature_important = pd.read_csv("./features_test.csv")

features_augment = feature_important[:5]
np.save("./data/lcld_augmented/important_features.npy", feature_important[:5]["Unnamed: 0"].to_numpy())
features_augment_mean = [np.mean(x_train[:, i]) for i in features_augment["Unnamed: 0"]]
x_train_augment = []
x_test_augment = []

comb = combinations(range(len(features_augment["Unnamed: 0"])), 2)

for i1, i2 in comb:
    x_train_augment.append(
        np.logical_xor(
            (x_train[:, features_augment["Unnamed: 0"][i1]] >= features_augment_mean[i1]),
            (x_train[:, features_augment["Unnamed: 0"][i2]] >= features_augment_mean[i2]),
        ).astype(np.float64)
    )
    x_test_augment.append(
        np.logical_xor(
            (x_test[:, features_augment["Unnamed: 0"][i1]] >= features_augment_mean[i1]),
            (x_test[:, features_augment["Unnamed: 0"][i2]] >= features_augment_mean[i2]),
        ).astype(np.float64)
    )

x_train_augment = np.column_stack(x_train_augment)
x_test_augment = np.column_stack(x_test_augment)

features_to_add = [
    {
        "feature": f"augmented_{i}",
        "type": "int",
        "mutable": True,
        "min": 0.0,
        "max": 1.0,
        "augmentation": True,
    }
    for i in range(x_train_augment.shape[1])
]
features_augmented = features.append(features_to_add)
constraints = pd.read_csv("./data/lcld/constraints.csv")
constraints_to_add = [
    {
        "min": 0.0,
        "max": 1.0,
        "augmentation": True,
    }
    for i in range(x_train_augment.shape[1])
]
constraints_augmented = constraints.append(constraints_to_add)

x_train_augment = np.concatenate((x_train, x_train_augment), axis=1)
x_test_augment = np.concatenate((x_test, x_test_augment), axis=1)
scaler = MinMaxScaler()
scaler.fit(np.concatenate((x_train_augment, x_test_augment)))


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


x_train_s = scaler.transform(x_train_augment)

model_robust = train_model(x_train_s, y_train)
y_proba = model_robust.predict_proba(scaler.transform(x_test_augment)).reshape(-1)
threshold = 0.25
y_pred = (y_proba >= threshold).astype(int)
print_score(y_test, y_pred)


# weak = train_model(x_train_s[:, features["augmentation"]], y_train)
#
# for i, model in enumerate([strong, weak]):
#     if i == 0:
#         y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
#     else:
#         y_proba = model.predict_proba(
#             scaler.transform(x_test)[:, features["augmentation"]]
#         ).reshape(-1)
#     mccs = [
#         matthews_corrcoef(y_test, (y_proba >= t / 100).astype(int)) for t in range(100)
#     ]
#     threshold = np.argmax(mccs) / 100
#     print(f"{threshold}")
#     y_pred = (y_proba >= threshold).astype(int)
#     print_score(y_test, y_pred)


#
candidates_index = (y_test == 1) * (y_test == y_pred)
# print(candidates_index.shape)
# print(candidates_index.sum())
X_candidates = x_test_augment[candidates_index, :]
X_candidates_lcld = np.load("./data/lcld/x_attack_candidates.npy")


def index_row_in_array(row, arr):
    return np.where(np.sum(np.abs(arr - row), axis=1) == 0)[0]


indexes = [
    index_row_in_array(row, X_candidates[:, :X_candidates_lcld.shape[1]])
    for row in tqdm(X_candidates_lcld[:4000], total=4000)
]
indexes = np.array([i[0] for i in indexes if len(i) > 0])
X_candidates = X_candidates[indexes]

print(X_candidates.shape)
#
np.save("./data/lcld_augmented/x_attack_candidates.npy", X_candidates)
# #
tf.keras.models.save_model(
    model_robust,
    "./models/lcld_augmented/nn.model",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)
joblib.dump(scaler, "./models/lcld_augmented/scaler.joblib")
np.save("./data/lcld_augmented/features_augment_mean.npy", features_augment_mean)
features_augmented.to_csv("./data/lcld_augmented/features.csv")
constraints_augmented.to_csv("./data/lcld_augmented/constraints.csv")
