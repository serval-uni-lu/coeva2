import pickle
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
seed(2305)
import tensorflow as tf
tf.random.set_seed(2405)
import seaborn as sns
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from utils import in_out
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


### Load Data

x_train = np.load("../data/malware/training_data/X_train.npy")
y_train = np.load("../data/malware/training_data/y_train.npy")
x_test = np.load("../data/malware/training_data/X_test.npy")
y_test = np.load("../data/malware/training_data/y_test.npy")
x_candidates = np.load("../data/malware/x_attack_candidate.npy")
df = pd.read_csv("../data/malware/features.csv")


# Train a first model
model = RandomForestClassifier(random_state=2005)
model.fit(x_train, y_train)
probas = model.predict_proba(x_test)
predictions = (probas[:, 1] >= 0.5).astype(int)

### Evaluatte it
print_score(y_test, predictions)

### Keep Only most important features

index_to_keep = np.arange(x_train.shape[1])[23528:]
col = df["feature"].str.startswith("imp") + df["feature"].str.endswith("dll")
col_delete = np.argwhere(col.to_numpy()).reshape(-1)
col_keep = np.argwhere(~col.to_numpy()).reshape(-1)
print(col_keep.shape)
print(col_delete.shape)
f = model.feature_importances_[col_delete]
take = 1000
f_ind = np.argsort(f)[-take:]
index_to_keep = np.sort(np.concatenate([col_keep, col_delete[f_ind]]))
print(index_to_keep.shape)

### Retrain a classifier with only the most important features
model = RandomForestClassifier(random_state=2005)
model.fit(x_train[:, index_to_keep], y_train)
probas = model.predict_proba(x_test[:, index_to_keep])
predictions = (probas[:, 1] >= 0.5).astype(int)

### Evaluate it
print_score(y_test, predictions)


### Save the data and model
np.save("../data/malware/training_data_small/index_kept", index_to_keep)
np.save("../data/malware/training_data_small/X_train.npy", x_train[:, index_to_keep])
np.save("../data/malware/training_data_small/X_test.npy", x_test[:, index_to_keep])
joblib.dump(model, "../models/malware/model_small.joblib")

### Find in original feature file the features to keep
df = df.iloc[index_to_keep]
print(df.shape)
df.to_csv("../data/malware/features_small.csv")

### Train a scaler
scaler = MinMaxScaler()
x_all = np.concatenate((x_train[:, index_to_keep], x_test[:, index_to_keep]))
x_min = df["min"]
x_max = df["max"]
x_min[x_min == "dynamic"] = np.min(x_all, axis=0)[x_min == "dynamic"]
x_max[x_max == "dynamic"] = np.max(x_all, axis=0)[x_max == "dynamic"]
x_min = x_min.astype(np.float).to_numpy().reshape(1, -1)
x_max = x_max.astype(np.float).to_numpy().reshape(1, -1)
x_min = np.min(np.concatenate((x_min, x_all)), axis=0).reshape(1, -1)
x_max = np.max(np.concatenate((x_max, x_all)), axis=0).reshape(1, -1)
scaler.fit(np.concatenate((np.floor(x_min), np.ceil(x_max))))
joblib.dump(scaler, "../models/malware/scaler_small.pickle")

### From the features, print information for constraints.

features = df["feature"]
features = features.reset_index()["feature"]

frequencies = features.str.startswith("freq_byte")
freq_idx = np.argwhere(frequencies.to_numpy()).reshape(-1).tolist()
print(freq_idx)
with open("../data/malware/freq_idx_small.pkl", "wb") as f:
    pickle.dump(freq_idx, f)

print("api_import_nb index to be replaced")
print(features[features == "api_import_nb"].index[0])

imports = features.str.startswith("imp_")
imports_idx = np.argwhere(imports.to_numpy()).reshape(-1).tolist()
print(imports_idx)
with open("../data/malware/imports_idx_small.pkl", "wb") as f:
    pickle.dump(imports_idx, f)


print("header_NumberOfSections import index to be replaced")
print(features[features == "header_NumberOfSections"].index[0])


section_names = features.str.startswith("pesection_") & features.str.endswith("name")
section_names_idx = np.argwhere(section_names.to_numpy()).reshape(-1).tolist()
print(section_names_idx)
with open("../data/malware/section_names_idx_small.pkl", "wb") as f:
    pickle.dump(section_names_idx, f)
#
print("dll_import_nb index to be replaced")
print(features[features == "dll_import_nb"].index[0])

dll_imports = features.str.endswith(".dll")
dll_imports_idx = np.argwhere(dll_imports.to_numpy()).reshape(-1).tolist()
print(dll_imports_idx)
with open("../data/malware/dll_imports_idx_small.pkl", "wb") as f:
    pickle.dump(dll_imports_idx, f)

man_col = [
    "header_NumberOfSections",
    "header_FileAlignment",
    "header_SectionAlignment",
    "api_import_nb",
    "dll_import_nb",
    "generic_fileEntropy",
]

for e in man_col:
    print(f"{e} = {features[features == e].index[0]}")

## Train a surrogate




# ----- Model Training

x_train_surrogate = scaler.transform(x_train[:, index_to_keep])
y_train_surrogate = (model.predict_proba(x_train_surrogate)[:, 1] >= 0.5).astype(int)


surrogate = Sequential()
surrogate.add(Dense(units=128, activation="relu", input_dim=x_train_surrogate.shape[1]))
surrogate.add(Dense(units=56, activation="relu"))
surrogate.add(Dense(units=28, activation="relu"))
surrogate.add(Dropout(0.2))
surrogate.add(Dense(1, activation='sigmoid'))
sgd = Adam(lr=0.001)
surrogate.compile(loss="binary_crossentropy", optimizer=sgd)

es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
r = surrogate.fit(
    x_train_surrogate,
    y_train_surrogate,
    validation_data=(x_train_surrogate, y_train_surrogate),
    epochs=400,
    batch_size=128,
    callbacks=[es],
)

y_proba_surrogate = surrogate.predict_proba(scaler.transform(x_test[:, index_to_keep]))
y_pred_surrogate = (y_proba_surrogate >= 0.5).astype(int).reshape(-1)
r2 = r2_score(probas[:, 1], y_proba_surrogate)
print(f"r2 score{r2}")
print("Score relative to model")
print_score(predictions, y_pred_surrogate)
print("Absolute score")
print_score(y_test, y_pred_surrogate)

# X_candidates

candidates_index = (y_test == predictions) * (y_test == 1) * (y_test == y_pred_surrogate)
print(candidates_index.shape)
print(candidates_index.sum())
X_candidate = x_test[:, index_to_keep][candidates_index]
print(X_candidate.shape)

# Save
tf.keras.models.save_model(
        surrogate,
        "../models/malware/surrogate_small.model",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
    )
np.save("../data/malware/x_attack_candidate_small.npy", X_candidate)

