import logging
import traceback
from pathlib import Path

import pandas as pd


def add_human_names(df: pd.DataFrame) -> pd.DataFrame:
    attack_name = {
        "moeva": "MoEvA2",
        "flip": "PGD",
        "constraints+flip+adaptive_eps_step+repair": "C-PGD",
        "papernot": "Iterative Papernot",
    }
    df["attack_name_human"] = df["attack_name"].map(attack_name)
    return df


def parse_moeva(metrics):
    config = metrics["config"]
    return [
        {
            "attack_name": config["attack_name"],
            "eps": config["eps_list"][i],
            **metrics["objectives_list"][i],
        }
        for i in range(len(metrics["objectives_list"]))
    ]


def parse_pgd(metrics):
    config = metrics["config"]
    return {
        "attack_name": config["loss_evaluation"],
        "eps": config["eps"],
        **metrics["objectives"],
    }


def parse_papernot(metrics):
    config = metrics["config"]
    return {
        "attack_name": config["attack_name"],
        "eps": config["eps"],
        **metrics["objectives"],
    }


def parse_metric(metric):
    try:
        config = metric["config"]
        project_name = config["project_name"].replace("_augmented", "")
        parsed = {
            "n_input": config["n_input"],
            "config_hash": metric["config_hash"],
            "project_name": project_name,
            "budget": config["budget"],
            "time": metric["time"],
            "model": Path(config["paths"]["model"]).stem,
            "reconstruction": config.get("reconstruction", None),
        }
        if metric["config"]["attack_name"] == "moeva":
            return [{**parsed, **parsed_moeva} for parsed_moeva in parse_moeva(metric)]
        elif metric["config"]["attack_name"] == "pgd":
            return [{**parsed, **parse_pgd(metric)}]
        elif metric["config"]["attack_name"] == "papernot":
            return [{**parsed, **parse_papernot(metric)}]
        else:
            logging.error(f"No parser for attack {metric['config']['attack_name']}")
            return []
    except:
        logging.error(f"Error with metric (wrong format?): {metric}.")
        traceback.print_exc()


def parse_metrics(metrics):
    out = []
    for metric in metrics:
        out.extend(parse_metric(metric))
    return out
