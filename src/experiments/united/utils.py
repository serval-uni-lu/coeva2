import joblib
from sklearn.base import ClassifierMixin
from tensorflow.keras import Sequential
from tensorflow.keras.models import save_model as tf_save_model

from src.constraints.botnet.botnet_constraints import (
    BotnetConstraints,
    BotnetAugmentedConstraints,
)
from src.constraints.botnet.botnet_constraints_sat import (
    create_constraints as botnet_sat_constraints,
)
from src.constraints.lcld.lcld_augmented_constraints import LcldAugmentedConstraints
from src.constraints.lcld.lcld_constraints import LcldConstraints
from src.constraints.lcld.lcld_constraints_sat import (
    create_constraints as lcld_sat_constraints,
)
from src.constraints.malware.malware_constraints import MalwareConstraints
from src.constraints.malware.malware_constraints_sat import (
    create_constraints as malware_sat_constraints,
)
from src.constraints.url.url_constraints import UrlConstraints, UrlAugmentedConstraints
from src.constraints.url.url_constraints_sat import (
    create_constraints as url_sat_constraints,
)
from src.datasets.malware_dataset import MalwareDataset

STR_TO_CONSTRAINTS_CLASS = {
    "lcld": LcldConstraints,
    "botnet": BotnetConstraints,
    "malware": MalwareConstraints,
    "lcld_augmented": LcldAugmentedConstraints,
    "botnet_augmented": BotnetAugmentedConstraints,
    "url": UrlConstraints,
    "url_rf": UrlConstraints,
    "url_augmented": UrlAugmentedConstraints,
    "url_rf_augmented": UrlAugmentedConstraints,
}

STR_TO_SAT_CONSTRAINTS = {
    "lcld": lcld_sat_constraints,
    "botnet": botnet_sat_constraints,
    "malware": malware_sat_constraints,
    "url": url_sat_constraints,
    "url_rf": url_sat_constraints,
}

STR_TO_DATASET = {"malware": MalwareDataset}


def get_constraints_from_str(project_name: str):
    return STR_TO_CONSTRAINTS_CLASS[project_name]


def get_sat_constraints_from_str(project_name: str):
    return STR_TO_SAT_CONSTRAINTS[project_name]


def get_dataset(name: str):
    return STR_TO_DATASET[name]


def save_model(model, model_path):
    if isinstance(model, Sequential):
        tf_save_model(model, model_path)
    elif isinstance(model, ClassifierMixin):
        joblib.dump(model, model_path)
    else:
        raise NotImplementedError
