from abc import ABC, abstractmethod
from typing import Sequence


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
    def get_best(self) -> Sequence[float]:
        raise NotImplementedError
