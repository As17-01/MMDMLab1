from abc import ABC
from abc import abstractmethod


class BaseGeneticAlgorithm(ABC):
    """Base class for genetic algorithm."""

    @abstractmethod
    def mutate(self, delta: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def select(self, keep_share: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def mate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_best(self):
        raise NotImplementedError
