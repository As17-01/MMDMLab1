from abc import ABC
from typing import List
from typing import Tuple

import numpy as np
from loguru import logger


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

        self._population = np.random.uniform(
            low=constraints[0],
            high=constraints[1],
            size=(population_size, dimension_size),
        ).tolist()


class CouriersGeneticAlgorithmState(BaseGeneticAlgorithmState):
    def __init__(
        self,
        capacity: List[int],
        demand: List[int],
        population_size: int,
        random_state: int,
    ):
        self._population_size = population_size
        self._num_couriers = len(capacity)
        self._capacity = capacity
        self._num_cities = len(demand)
        self._demand = demand
        self._random_state = random_state

        np.random.seed(random_state)

        self._population = []
        average_num_retries = 0
        while len(self._population) < self._population_size:

            new_pop = self._generate_distribution()
            while not self._validate_all_present(new_pop):
                average_num_retries += 1 / self._population_size
                new_pop = self._generate_distribution()

            if new_pop in self._population:
                continue
            self._population.append(new_pop)

        logger.info(f"Average Num retries: {average_num_retries}")
        self._population = self._population

    def _validate_all_present(self, distribution: List[List[int]]) -> bool:
        cities = np.arange(1, self._num_cities)
        for city in cities:
            is_present = 0
            for route in distribution:
                if city in route:
                    is_present += 1
            if is_present == 0:
                return False
        return True

    def _validate_capacity(self, distribution: List[List[int]]) -> bool:
        for courier_id, courier_route in enumerate(distribution):
            if len(courier_route) > 0:
                if np.sum(np.array(self._demand)[np.array(courier_route)]) > self._capacity[courier_id]:
                    return False
        return True

    def _generate_distribution(self) -> List[List[int]]:
        cities = np.arange(1, self._num_cities)
        np.random.shuffle(cities)

        distribution = [[] for _ in range(self._num_couriers)]
        courier_load = np.zeros(self._num_couriers)
        for city in cities:
            for courier_id in range(self._num_couriers):
                if courier_load[courier_id] + self._demand[city] <= self._capacity[courier_id]:
                    courier_load[courier_id] += self._demand[city]
                    distribution[courier_id].append(city)
                    break

        if not self._validate_capacity(distribution):
            raise ValueError("Generated an invalid distribution!")

        return distribution
