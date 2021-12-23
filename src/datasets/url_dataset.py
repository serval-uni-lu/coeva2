import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .directory_dataset import DirectoryDataset


class UrlDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/url"
        super().__init__(dir_path)
