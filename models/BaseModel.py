from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Empty base class to define structure of models and necessary functions.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

