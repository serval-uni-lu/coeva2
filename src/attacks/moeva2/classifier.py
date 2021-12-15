import joblib
import numpy as np
import warnings

from src.utils.in_out import load_model


class Classifier:

    """
    Wrapper for classifier having a predict_proba method.
    Extend and override for other classifiers.
    """

    def __init__(self, classifier, n_jobs=1, verbose=0) -> None:
        if hasattr(classifier, "predict_proba") and callable(
            getattr(classifier, "predict_proba")
        ):
            self._classifier = classifier
        else:
            raise ValueError(
                "The provided model does not have methods 'predict_proba'."
            )
        self.set_n_jobs(n_jobs)
        self.set_verbose(verbose)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self._classifier.predict_proba(x)
        if proba.shape[1] == 1:
            proba = np.concatenate((1 - proba, proba), axis=1)
        return proba

    def set_verbose(self, verbose: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(
            getattr(self._classifier, "set_params")
        ):
            self._classifier.set_params(verbose=verbose)

    def set_n_jobs(self, n_jobs: int) -> None:
        if hasattr(self._classifier, "set_params") and callable(
            getattr(self._classifier, "set_params")
        ):
            self._classifier.set_params(n_jobs=n_jobs)


class DelayedClassifier:
    def __init__(self, classifier_path, n_jobs=1, verbose=0):
        self.classifier = None
        self.classifier_path = classifier_path

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.load()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.classifier.predict_proba(x)
        if proba.shape[1] == 1:
            proba = np.concatenate((1 - proba, proba), axis=1)
        return proba

    def load(self):
        if self.classifier is None:
            self.classifier = load_model(self.classifier_path)


class ScalerClassifier(DelayedClassifier):
    def __init__(self, classifier_path, scaler_path, n_jobs=1, verbose=0):
        super().__init__(classifier_path, n_jobs, verbose)
        self.scaler = joblib.load(scaler_path)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(x)
        return super(ScalerClassifier, self).predict_proba(x)
