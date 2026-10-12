from abc import ABC, abstractmethod

from value import Value


class Model(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def create_dataset(self):
        ...

    @abstractmethod
    def calc_loss(self):
        ...

    @abstractmethod
    def gradient_descent(self):
        ...
