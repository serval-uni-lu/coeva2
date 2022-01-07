from src.datasets.botnet_dataset import BotnetDataset, BotnetAugmentedDataset
from src.datasets.malware_dataset import (
    MalwareDataset,
    MalwareAugmentedDataset,
    MalwareRfAugmentedDataset,
    MalwareRfDataset,
)
from src.datasets.url_dataset import (
    UrlDataset,
    UrlAugmentedDataset,
    UrlRfAugmentedDataset,
)


datasets = {
    "url": UrlDataset,
    "url_augmented": UrlAugmentedDataset,
    "url_rf": UrlDataset,
    "url_rf_augmented": UrlRfAugmentedDataset,
    "botnet": BotnetDataset,
    "botnet_augmented": BotnetAugmentedDataset,
    "malware": MalwareDataset,
    "malware_augmented": MalwareAugmentedDataset,
    "malware_rf": MalwareRfDataset,
    "malware_rf_augmented": MalwareRfAugmentedDataset,
}


def load_dataset(name: str):
    if name in datasets:
        return datasets[name]()
    else:
        raise Exception("Object '%s' for not found in %s" % (name, datasets))
