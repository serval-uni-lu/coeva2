import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping

np.random.seed(2505)
import tensorflow as tf

tf.random.set_seed(2605)
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, accuracy_score

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



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


def create_DNN(units, input_dim_param, lr_param):
    network = Sequential()
    network.add(Dense(units=units[0], activation="relu", input_dim=input_dim_param))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[1], activation="relu"))
    network.add(Dropout(0.1))
    network.add(Dense(units=units[2], activation="relu"))
    network.add(Dense(units=1, activation="sigmoid"))

    sgd = Adam(learning_rate=lr_param)
    network.compile(loss="binary_crossentropy", optimizer=sgd)

    return network


x_train = np.load("./data/botnet/train_test/x_train.npy")
y_train = np.load("./data/botnet/train_test/y_train.npy")
x_test = np.load("./data/botnet/train_test/x_test.npy")
y_test = np.load("./data/botnet/train_test/y_test.npy")
scaler = MinMaxScaler()
df = pd.read_csv("./data/botnet/features_ctu.csv")

x_all = np.concatenate((x_train, x_test))
x_min = df["min"]
x_max = df["max"]
x_min[x_min == "dynamic"] = np.min(x_all, axis=0)[x_min == "dynamic"]
x_max[x_max == "dynamic"] = np.max(x_all, axis=0)[x_max == "dynamic"]
x_min = x_min.astype(np.float).to_numpy().reshape(1, -1)
x_max = x_max.astype(np.float).to_numpy().reshape(1, -1)
x_min = np.min(np.concatenate((x_min, x_all)), axis=0).reshape(1, -1)
x_max = np.max(np.concatenate((x_max, x_all)), axis=0).reshape(1, -1)
scaler.fit(np.concatenate((np.floor(x_min), np.ceil(x_max))))

joblib.dump(scaler, "./models/botnet/scaler.joblib")


LAYERS = [32, 16, 8]
INPUT_DIM = 756
LR = [0.001]

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)

nn = create_DNN(units=LAYERS, input_dim_param=INPUT_DIM, lr_param=LR[0])

nn.fit(scaler.transform(x_train), y_train, verbose=1, epochs=20, batch_size=64, shuffle=True, callbacks=[es], validation_split=0.1,)

y_proba = nn.predict_proba(scaler.transform(x_test)).reshape(-1)
y_pred = (y_proba >= 0.5).astype(int)


print_score(y_test, y_pred)

candidates_index = (
    (y_test == 1) * (y_test == y_pred)
)
print(candidates_index.shape)
print(candidates_index.sum())
X_candidate = x_test[candidates_index, :]
print(X_candidate.shape)

np.save("./data/botnet/x_attack_candidate_small.npy", X_candidate)

tf.keras.models.save_model(
    nn,
    "./models/botnet/nn.model",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)