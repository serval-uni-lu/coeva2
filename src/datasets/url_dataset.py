from .directory_dataset import DirectoryDataset


class UrlDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/url"
        super().__init__(dir_path)


class UrlAugmentedDataset(DirectoryDataset):
    def __init__(self):
        dir_path = "./data/url_augmented"
        super().__init__(dir_path)
