from sklearn.ensemble import RandomForestClassifier

from src.models.botnet_model import BotnetModel
from src.models.malware_model import MalwareModel
from src.models.model_architecture import ModelArchitecture
from src.models.url_model import UrlModel


class DefaultRf(ModelArchitecture):
    def get_model(self):
        model = RandomForestClassifier(n_estimators=100)
        return model

    def get_trained_model(self, X, y):
        model = self.get_model()
        model.set_params(**{"verbose": 1})
        model.fit(X, y)
        model.set_params(**{"verbose": 0})
        return model


model_architectures = {
    "url": UrlModel(),
    "url_rf": DefaultRf(),
    "botnet": BotnetModel(),
    "botnet_rf": DefaultRf(),
    "malware": MalwareModel(),
    "malware_rf": DefaultRf(),
}


def load_model_architecture(name: str) -> ModelArchitecture:
    if name in model_architectures:
        return model_architectures[name]
    else:
        raise Exception("Object '%s' for not found in %s" % (name, model_architectures))
