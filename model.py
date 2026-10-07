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
    def plot_points(self):
        ...

    @abstractmethod
    def calc_loss(self) -> tuple[Value, float]:
        ...

    @abstractmethod
    def gradient_descent(self):
        ...
