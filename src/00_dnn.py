import joblib
import numpy as np
from src.examples.lcld.lcld_constraints import LcldConstraints
from utils import in_out
from tensorflow.keras.models import load_model

config = in_out.get_parameters()




def run(
    SURROGATE_PATH=config["paths"]["surrogate"],
    SCALER_PATH=config["paths"]["scaler"],
    THRESHOLD=config["threshold"],
    X_SURROGATE_CANDIDATES_PATH=config["paths"]["x_surrogate_candidates"],
):

    model = load_model(SURROGATE_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test = np.load(config["paths"]["x_candidates"])
    y_pred_proba = model.predict(scaler.fit_transform(X_test))
    y_pred = (y_pred_proba[:, 1] >= THRESHOLD).astype(bool)

    X_test = X_test[y_pred]

    # Removing x that violates constraints
    constraints_evaluator = LcldConstraints(
        # config["amount_feature_index"],
        config["paths"]["features"],
        config["paths"]["constraints"],
    )
    constraints = constraints_evaluator.evaluate(X_test)
    # Scaling tolerance = 5
    constraints[:, 0] = constraints[:, 0] - 5
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_test = X_test[(1 - constraints_violated).astype(bool)]
    print("{} candidates.".format(X_test.shape[0]))
    np.save(X_SURROGATE_CANDIDATES_PATH, X_test)
    print("Hello.")

if __name__ == "__main__":
    run()
