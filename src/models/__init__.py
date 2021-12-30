from src.models.malware_model import MalwareModel
from src.models.model_architecture import ModelArchitecture
from src.models.url_model import UrlModel

model_architectures = {
    "url": UrlModel(),
    "malware": MalwareModel()
}


def load_model_architecture(name: str) -> ModelArchitecture:
    if name in model_architectures:
        return model_architectures[name]
    else:
        Exception("Object '%s' for not found in %s" % (name, model_architectures))
