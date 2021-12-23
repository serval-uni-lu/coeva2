import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .abstract_dataset import Dataset


class DirectoryDataset(Dataset):
    def __init__(self, dir_path):
        self.X_train = np.load(f"{dir_path}/X_train.npy")
        self.y_train = np.load(f"{dir_path}/y_train.npy")
        self.X_test = np.load(f"{dir_path}/X_test.npy")
        self.y_test = np.load(f"{dir_path}/y_test.npy")
        self.std_scaler = None
        self.min_max_scaler = None

    def get_train_test(self) -> (np.ndarray, np.ndarray, np.array, np.ndarray):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _get_std_scaler(self):
        if self.std_scaler is None:
            self.std_scaler = StandardScaler()
            X_train, _, _, _ = self.get_train_test()
            self.std_scaler.fit(X_train)
        return self.std_scaler

    def _get_min_max_scaler(self):
        if self.min_max_scaler is None:
            self.min_max_scaler = MinMaxScaler()
            X_train, X_test, _, _ = self.get_train_test()
            X = np.concatenate([X_train, X_test])
            self.min_max_scaler.fit(X)
        return self.min_max_scaler

    def get_scaler(self, method="min_max"):
        if method == "std":
            return self._get_std_scaler()
        elif method == "min_max":
            return self._get_min_max_scaler()
        else:
            raise NotImplementedError
