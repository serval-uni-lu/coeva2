from src.datasets.malware_dataset import MalwareDataset
from src.constraints.botnet.botnet_augmented_constraints import BotnetAugmentedConstraints
from src.constraints.botnet.botnet_constraints import BotnetConstraints
from src.constraints.lcld.lcld_augmented_constraints import LcldAugmentedConstraints
from src.constraints.lcld.lcld_constraints import LcldConstraints
from src.constraints.malware.malware_constraints import MalwareConstraintsFast
from src.constraints.lcld.lcld_constraints_sat import (
    create_constraints as lcld_sat_constraints,
)
from src.constraints.botnet.botnet_constraints_sat import (
    create_constraints as botnet_sat_constraints,
)
from src.constraints.malware.malware_constraints_sat import (
    create_constraints as malware_sat_constraints,
)
from src.constraints.url.url_constraints import UrlConstraints

STR_TO_CONSTRAINTS_CLASS = {
    "lcld": LcldConstraints,
    "botnet": BotnetConstraints,
    "malware": MalwareConstraintsFast,
    "lcld_augmented": LcldAugmentedConstraints,
    "botnet_augmented": BotnetAugmentedConstraints,
    "url": UrlConstraints
}

STR_TO_SAT_CONSTRAINTS = {
    "lcld": lcld_sat_constraints,
    "botnet": botnet_sat_constraints,
    "malware": malware_sat_constraints,
}

STR_TO_DATASET = {
    "malware": MalwareDataset
}


def get_constraints_from_str(project_name: str):
    return STR_TO_CONSTRAINTS_CLASS[project_name]


def get_sat_constraints_from_str(project_name: str):
    return STR_TO_SAT_CONSTRAINTS[project_name]

def get_dataset(name: str):
    return STR_TO_DATASET[name]