from typing import Tuple

import numpy as np


from abc import ABC


class BaseGeneticAlgorithmState(ABC):
    """Base class for genetic algorithm state."""

    @property
    def population(self):
        return self._population
    
    @population.setter
    def population(self, new_population):
        self._population = new_population

    @property
    def population_size(self):
        return self._population_size


class DotsGeneticAlgorithmState(BaseGeneticAlgorithmState):
    def __init__(
        self,
        population_size: int,
        dimension_size: int,
        constraints: Tuple[float, float],
        random_state: int,
    ):
        self._population_size = population_size
        self._dimension_size = dimension_size
        self._constraints = constraints
        self._random_state = random_state

        np.random.seed(random_state)

        self._population = list(np.random.uniform(
            low=constraints[0],
            high=constraints[1],
            size=(population_size, dimension_size),
        ))


class CouriersGeneticAlgorithmState(BaseGeneticAlgorithmState):
    def __init__(
        self,
        num_couriers: int, 
        num_cities: int, 
        population_size: int,
        random_state: int,
        
    ):
        self._population_size = population_size
        self._num_couriers = num_couriers
        self._num_cities = num_cities
        self._random_state = random_state

        np.random.seed(random_state)

        if self._num_cities < self._num_couriers:
            raise ValueError("Количество пунктов должно быть не меньше количества курьеров.")

        self._population = []
        for _ in range(self._population_size):
            self._population.append(self._generate_distibution())

        self._population = self._population

    def _generate_distibution(self):
        cities = np.arange(1, self._num_cities + 1)
        np.random.shuffle(cities)

        div = np.sort(np.random.choice(np.arange(self._num_cities), self._num_couriers - 1, replace=False))
        distribution = np.split(cities, div)
        return distribution
