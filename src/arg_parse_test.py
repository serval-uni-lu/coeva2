import argparse
import json
import os
import yaml
from mergedeep import merge, Strategy
import re
from src.config_parser.config_parser import get_config

file_parsers = {".yaml": yaml.full_load, ".yml": yaml.full_load, ".json": json.load}


def file_to_dict(path):
    extension = os.path.splitext(path)[1]
    with open(path, "r") as f:
        return file_parsers[extension](f)


def value_parser(value):
    SPECIAL_KEY = "SPECIAL_KEY"
    if re.match(r"^-?\d+(?:\.\d+)$", value) is None:
        return str(value)
    else:
        return yaml.safe_load(f"{SPECIAL_KEY}: {value}")[SPECIAL_KEY]


def str_to_dict(key, value):
    splits = key.split(".", maxsplit=1)
    current_key = splits[0]
    next_key = splits[1] if len(splits) > 1 else None
    if next_key is None:
        dictionary = {current_key: value}
    else:
        dictionary = {current_key: str_to_dict(next_key, value)}
    return dictionary


def merge_parameters(a, b):
    return merge(a, b, strategy=Strategy.REPLACE)


def parse_arguments(arguments):

    current_parameters = {}
    if arguments.c is not None:
        for config_file in arguments.c:
            incoming_parameters = file_to_dict(config_file)
            merge_parameters(current_parameters, incoming_parameters)

    if arguments.j is not None:
        for parameter in arguments.j:
            incoming_parameters = json.loads(str(parameter))
            merge_parameters(current_parameters, incoming_parameters)

    if arguments.p is not None:
        for parameter in arguments.p:
            splits = str(parameter).split("=", maxsplit=1)
            key, value = splits[0], value_parser(splits[1])
            incoming_parameters = str_to_dict(key, value)
            merge_parameters(current_parameters, incoming_parameters)

    return current_parameters


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    help="Provide config file in yaml or json.",
    action="append",
)
parser.add_argument(
    "-p",
    help="Provide extra parameters on the form key1.key2[key3]=value.",
    action="append",
)
print(parser.add_argument(
    "-j",
    help="Inline json.",
    action="append",
).dest)
args = vars(parser.parse_args())
print(args["c"])
# print(parse_arguments(args))

print(get_config())