from typing import Tuple

import numpy as np


class DotsGeneticAlgorithmState:
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
        )

    @property
    def population(self):
        return self._population
    
class CouriersGeneticAlgorithmState:
    def __init__(
        self,
        num_couriers: int, 
        num_points: int, 
        population_size: int,
        random_state: int = None
        
    ):
        self.num_couriers = num_couriers
        self.num_points = num_points
        self.population_size = population_size
        self.random_state = random_state

        np.random.seed(random_state)

        if self.num_points < self.num_couriers:
            raise ValueError("Количество пунктов должно быть не меньше количества курьеров.")

        self._population = []
        for _ in range(self.population_size):

            points = np.random.choice(self.num_points, self.num_points, replace=False)
            
            distribution = [[point] for point in points[:self.num_couriers]]

            remaining_points = points[self.num_couriers:]
            for point in remaining_points:
                distribution[np.random.randint(0, self.num_couriers)].append(point)

            self._population.append(distribution)

    @property
    def population(self):
        return self._population
