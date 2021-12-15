import numpy as np

from .abstract_dataset import Dataset


class DirectoryDataset(Dataset):
    def __init__(self, dir_path):
        self.X_train = np.load(f"{dir_path}/X_train.npy")
        self.y_train = np.load(f"{dir_path}/y_train.npy")
        self.X_test = np.load(f"{dir_path}/X_test.npy")
        self.y_test = np.load(f"{dir_path}/y_test.npy")

    def get_train_test(self) -> (np.ndarray, np.ndarray, np.array, np.ndarray):
        return self.X_train, self.X_test, self.y_train, self.y_test
