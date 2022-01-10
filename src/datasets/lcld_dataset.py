from .directory_dataset import DirectoryDataset
from .id_scaler import IdScaler

class LcldDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/lcld"
        super().__init__(dir_path)


class LcldAugmentedDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/lcld_augmented"
        super().__init__(dir_path)


class LcldRfAugmentedDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/lcld_rf_augmented"
        super().__init__(dir_path)
