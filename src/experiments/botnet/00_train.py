import numpy as np

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
scaler.fit(np.concatenate((x_train, x_test)))
joblib.dump(scaler, "./models/botnet/scaler.joblib")


LAYERS = [256, 128, 64]
INPUT_DIM = 756
LR = [0.0002592943797404667]

nn = create_DNN(units=LAYERS, input_dim_param=INPUT_DIM, lr_param=LR[0])

nn.fit(scaler.transform(x_train), y_train, verbose=1, epochs=50, batch_size=64, shuffle=True)

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