import numpy as np
from imblearn.combine import SMOTEENN

np.random.seed(3105)
import tensorflow as tf
import joblib

tf.random.set_seed(3205)
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    EditedNearestNeighbours,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import EarlyStopping

kf = StratifiedKFold(n_splits=5, random_state=3105, shuffle=True)


x_train = np.load("./data/lcld/X_train.npy")
x_test = np.load("./data/lcld/X_test.npy")
y_train = np.load("./data/lcld/y_train.npy")
y_test = np.load("./data/lcld/y_test.npy")

scaler = MinMaxScaler()
scaler.fit(np.concatenate((x_train, x_test)))

x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)


# def focal_loss(gamma=2.0, alpha=0.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(
#             alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1 + K.epsilon())
#         ) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0 + K.epsilon()))
#
#     return focal_loss_fixed


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
    # model = Sequential()
    # model.add(Dense(units=64, activation="relu"))
    # model.add(Dense(units=32, activation="relu"))
    # model.add(Dense(units=32, activation="relu"))
    # model.add(Dense(units=1, activation="sigmoid"))
    #
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model = Sequential()

    # input layer
    model.add(Dense(64, activation="relu"))
    # model.add(Dropout(0.2))
    #
    # hidden layer
    model.add(Dense(32, activation="relu"))
    # model.add(Dropout(0.2))
    #
    # hidden layer
    model.add(Dense(16, activation="relu"))
    # model.add(Dropout(0.2))

    # output layer
    model.add(Dense(1, activation="sigmoid"))

    # compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
    )

    return model


avg_mcc = []

# for train_index, test_index in kf.split(x_train, y_train):
#     model = create_model()
#
#     x_train_local, x_test_local = x_train_s[train_index], x_train_s[test_index]
#     y_train_local, y_test_local = y_train[train_index], y_train[test_index]
#
#     early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
#
#     x_train_local, x_val_local, y_train_local, y_val_local = train_test_split(
#         x_train_local,
#         y_train_local,
#         test_size=0.1,
#         random_state=42,
#         stratify=y_train_local,
#     )
#
#     # sampler = RandomUnderSampler(
#     #     random_state=42,
#     #     # sampling_strategy={0: 20000, 1: 20000}
#     # )
#     # print(x_train_local.shape)
#     # print(y_train_local.shape)
#     # x_train_local, y_train_local = sampler.fit_resample(x_train_local, y_train_local)
#     # print(x_train_local.shape)
#
#     # sampler = SMOTEENN(n_jobs=10)
#     #
#     # print(x_train_local.shape)
#     # print(y_train_local.shape)
#     # x_train_local, y_train_local = sampler.fit_resample(x_train_local, y_train_local)
#     # print(x_train_local.shape)
#
#     model.fit(
#         x=x_train_local,
#         y=y_train_local,
#         epochs=100,
#         batch_size=512,
#         # validation_split=0.1,
#         validation_data=(x_val_local, y_val_local),
#         # shuffle=True,
#         verbose=1,
#         # class_weight={0: 1, 1: 1.5},
#         callbacks=[early_stop],
#     )
#
#     y_proba = model.predict_proba(x_test_local).reshape(-1)
#     y_pred = (y_proba >= 0.22).astype(int)
#
#     mccs = [
#         matthews_corrcoef(y_test_local, (y_proba >= t / 100).astype(int))
#         for t in range(100)
#     ]
#
#     avg_mcc.append(np.max(mccs))
#     print(np.argmax(mccs))
#
#     print(np.max(mccs))
#
#     from sklearn.metrics import precision_recall_curve
#     from sklearn.metrics import plot_precision_recall_curve
#     import matplotlib.pyplot as plt
#
#     precision, recall, thresholds = precision_recall_curve(y_test_local, y_proba)
#     plt.plot(thresholds, precision[:-1], label="Precision")
#     plt.plot(thresholds, recall[:-1], label="Recall")
#     plt.plot(precision[:-1], recall[:-1], label="Precision recall")
#     plt.legend()
#     plt.show()
#
#     print_score(y_test_local, y_pred)

# avg_mcc = np.array(avg_mcc)
# print(avg_mcc.mean())


model = create_model()

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

x_train_local, x_val_local, y_train_local, y_val_local = train_test_split(
    x_train_s,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,
)

print(x_train_local.shape)
model.fit(
    x=x_train_local,
    y=y_train_local,
    epochs=100,
    batch_size=512,
    # validation_split=0.1,
    validation_data=(x_val_local, y_val_local),
    # shuffle=True,
    verbose=1,
    # class_weight={0: 1, 1: 1.5},
    callbacks=[early_stop],
)

y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
y_pred = (y_proba >= 0.2).astype(int)

print_score(y_test, y_pred)

candidates_index = (
    (y_test == 1) * (y_test == y_pred)
)
print(candidates_index.shape)
print(candidates_index.sum())
X_candidate = x_test[candidates_index, :]
print(X_candidate.shape)

np.save("./data/lcld/x_attack_candidate.npy", X_candidate)

tf.keras.models.save_model(
    model,
    "./models/lcld/nn.model",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)
joblib.dump(scaler, "./models/lcld/scaler.joblib")
