import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x_train = np.load("../data/malware/training_data/X_train.npy")
y_train = np.load("../data/malware/training_data/y_train.npy")
x_test = np.load("../data/malware/training_data/X_test.npy")
y_test = np.load("../data/malware/training_data/y_test.npy")
x_candidates = np.load("../data/malware/x_attack_candidate.npy")

df = pd.read_csv("../data/malware/features.csv")

index_to_keep = np.arange(x_train.shape[1])[23528:]
col = df["feature"].str.startswith("imp") + df["feature"].str.endswith("dll")
col_delete = np.argwhere(col.to_numpy()).reshape(-1)
col_keep = np.argwhere(~col.to_numpy()).reshape(-1)
print(col_keep.shape)
print(col_delete.shape)

model = RandomForestClassifier(random_state=2005)

model.fit(x_train, y_train)

probas = model.predict_proba(x_test)
predictions = (probas[:, 1] >= 0.5).astype(int)

print("F1", f1_score(y_test, predictions))
print("Accuracy", accuracy_score(y_test, predictions))
print("mcc", matthews_corrcoef(y_test, predictions))

f = model.feature_importances_[col_delete]
take = 1000
f_ind = np.argsort(f)[-take:]

index_to_keep = np.sort(np.concatenate([col_keep, col_delete[f_ind]]))
print(index_to_keep.shape)


model = RandomForestClassifier(random_state=2005)

model.fit(x_train[:, index_to_keep], y_train)


probas = model.predict_proba(x_test[:, index_to_keep])
predictions = (probas[:, 1] >= 0.5).astype(int)



print("F1", f1_score(y_test, predictions))
print("Accuracy", accuracy_score(y_test, predictions))
print("mcc", matthews_corrcoef(y_test, predictions))


np.save("../data/malware/training_data_small/index_kept", index_to_keep)
np.save("../data/malware/training_data_small/X_train.npy", x_train[:, index_to_keep])
np.save("../data/malware/training_data_small/X_test.npy", x_test[:, index_to_keep])

df = df.iloc[index_to_keep]
print(df.shape)
df.to_csv("../data/malware/features_small.csv")
joblib.dump(model, "../models/malware/model_small.joblib")
scaler = MinMaxScaler()
scaler.fit(x_train[:, index_to_keep])
joblib.dump(scaler, "../models/malware/scaler_small.pickle")

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
