import numpy as np


class Classifier:

    def __init__(self, classifier) -> None:
        if hasattr(classifier, "predict_proba") and callable(getattr(classifier, "predict_proba")):
            self._classifier = classifier
        else:
            raise ValueError("The provided model does not have methods `predict_proba`.")

    def predict_proba(self, x) -> np.ndarray:
        return self._classifier.predict_proba(x)

    def set_verbose(self, verbose: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(getattr(self._classifier, "set_params")):
            self._classifier.set_params(verbose=0)

    def set_n_jobs(self, n_jobs: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(getattr(self._classifier, "set_params")):
            self._classifier.set_params(n_jobs=n_jobs)
