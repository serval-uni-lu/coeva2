import abc


class ModelArchitecture(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_trained_model(self, X, y):
        raise NotImplementedError
