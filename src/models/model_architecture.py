import abc


class ModelArchitecture(abc.ABC, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError