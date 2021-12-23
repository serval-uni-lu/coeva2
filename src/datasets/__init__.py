from src.datasets.malware_dataset import MalwareDataset
from src.datasets.url_dataset import UrlDataset

datasets = {
    "url": UrlDataset(),
    "malware": MalwareDataset()
}


def load_dataset(name: str):
    if name in datasets:
        return datasets[name]
    else:
        Exception("Object '%s' for not found in %s" % (name, datasets))
