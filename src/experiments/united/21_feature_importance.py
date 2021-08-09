import logging

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.python.keras.models import load_model

from src.config_parser import config_parser
from src.utils.time import timing


@timing
def run():
    model = load_model(config["paths"]["model"])
    scaler = joblib.load(config["paths"]["min_max_scaler"])
    x_train = np.load(f"./data/{config.get('project_name')}/X_train.npy")
    y_train = np.load(f"./data/{config.get('project_name')}/y_train.npy")
    x_train_s = scaler.transform(x_train)

    sampler = RandomUnderSampler(sampling_strategy={0: 5000, 1: 5000}, random_state=42)
    x_train_s, y_train = sampler.fit_resample(x_train_s, y_train)

    explainer = shap.DeepExplainer(model, x_train_s)
    features = pd.read_csv(config["paths"]["features"])
    shap_values = explainer.shap_values(x_train_s)
    
    shap_values_per_features = np.mean(np.abs(np.array(shap_values)[0]), axis=0)
    order_feature = np.argsort(shap_values_per_features)

    features['importance'] = shap_values_per_features
    df2 = features[features["mutable"]].sort_values(["importance"], ascending=False)
    print(df2)
    df2.to_csv("./features_test.csv")
    
    # print(features["feature"].to_numpy()[np.argsort(shap_values)])
    shap.summary_plot(
        shap_values,
        features=x_train_s,
        feature_names=features["feature"],
        plot_type="bar",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    config = config_parser.get_config()
    run()
