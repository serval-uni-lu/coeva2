import numpy as np

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
from tensorflow.python.keras.callbacks import EarlyStopping


x_train = np.load("./data/lcld/X_train.npy")
x_test = np.load("./data/lcld/X_test.npy")
y_train = np.load("./data/lcld/y_train.npy")
y_test = np.load("./data/lcld/y_test.npy")
x_retrain = np.load("./data/lcld/x_retrain_moeva.npy")
y_retrain = np.load("./data/lcld/y_retrain_moeva.npy")
x_candidates = np.load('./data/lcld/x_attack_candidates.npy')

scaler = joblib.load("./models/lcld/scaler.joblib")
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)
x_retrain_s = scaler.transform(x_retrain)



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


avg_mcc = []


model = create_model()

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

x_train_local, x_val_local, y_train_local, y_val_local = train_test_split(
    x_train_s,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,)


x_train_local = np.concatenate((x_train_local, x_retrain_s), axis=0)
y_train_local = np.concatenate((y_train_local, y_retrain), axis=0)

print(x_train_local.shape)
model.fit(
    x=x_train_local,
    y=y_train_local,
    epochs=100,
    batch_size=512,
    validation_data=(x_val_local, y_val_local),
    # shuffle=True,
    verbose=1,
    callbacks=[early_stop],
)

y_proba = model.predict_proba(scaler.transform(x_test)).reshape(-1)
mccs = [matthews_corrcoef(y_test, (y_proba >= t / 100).astype(int)) for t in range(100)]

threshold = 0.25
# print(threshold)
y_pred = (y_proba >= threshold).astype(int)

print_score(y_test, y_pred)

# From the candidates take second half and check
x_candidates = x_candidates[int(x_candidates.shape[0]/2):]
y_proba = model.predict_proba(scaler.transform(x_candidates)).reshape(-1)
y_pred = (y_proba >= threshold).astype(int)


candidates_index_retrain = (1 == y_pred)

print(candidates_index_retrain.shape)
print(candidates_index_retrain.sum())
X_candidate_retrain = x_candidates[candidates_index_retrain, :]
print(X_candidate_retrain.shape)

np.save("./data/lcld/x_attack_candidates_retrain.npy", X_candidate_retrain)

tf.keras.models.save_model(
    model,
    "./models/lcld/nn_retrained.model",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)
