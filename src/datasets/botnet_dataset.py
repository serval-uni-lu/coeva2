from .directory_dataset import DirectoryDataset


class BotnetDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/botnet"
        super().__init__(dir_path)


class BotnetAugmentedDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/botnet_augmented"
        super().__init__(dir_path)

class BotnetRfAugmentedDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/botnet_rf_augmented"
        super().__init__(dir_path)
