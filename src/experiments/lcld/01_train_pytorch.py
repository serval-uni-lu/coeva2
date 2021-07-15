import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Data
from sklearn.preprocessing import MinMaxScaler

x_train = np.load("./data/lcld/X_train.npy")
x_test = np.load("./data/lcld/X_test.npy")
y_train = np.load("./data/lcld/y_train.npy")
y_test = np.load("./data/lcld/y_test.npy")


scaler = MinMaxScaler()
scaler.fit(np.concatenate((x_train, x_test)))

x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)



epochs, set_size, batch_size, input_dim, hidden_dim, out_dim = (
    5,
    x_train_s.shape[0],
    32,
    x_train_s.shape[1],
    [64, 32, 16],
    1,
)
LEARNING_RATE = 1e-5

model = torch.nn.Sequential(
    *[
        torch.nn.Linear(input_dim, i)
        if index == 0
        else torch.nn.Linear(hidden_dim[index - 1], i)
        for (index, i) in enumerate(hidden_dim)
    ],
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim[-1], out_dim),
)
BATCH_TEST = 1024

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)