from src.datasets.botnet_dataset import BotnetDataset, BotnetAugmentedDataset
from src.datasets.malware_dataset import MalwareDataset, MalwareAugmentedDataset
from src.datasets.url_dataset import UrlDataset, UrlAugmentedDataset

datasets = {
    "url": UrlDataset,
    "url_augmented": UrlAugmentedDataset,
    "botnet": BotnetDataset,
    "botnet_augmented": BotnetAugmentedDataset,
    "malware": MalwareDataset,
    "malware_augmented": MalwareAugmentedDataset
}


def load_dataset(name: str):
    if name in datasets:
        return datasets[name]()
    else:
        raise Exception("Object '%s' for not found in %s" % (name, datasets))
