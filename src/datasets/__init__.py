from src.datasets.malware_dataset import MalwareDataset
from src.datasets.url_dataset import UrlDataset, UrlAugmentedDataset

datasets = {
    "url": UrlDataset,
    "url_augmented": UrlAugmentedDataset,
    "malware": MalwareDataset
}


def load_dataset(name: str):
    if name in datasets:
        return datasets[name]()
    else:
        raise Exception("Object '%s' for not found in %s" % (name, datasets))
