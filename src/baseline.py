from typing import Callable, Sequence

import numpy as np

from src.base import BaseGeneticAlgorithm
from src.state import GeneticAlgorithmState


class BaselineGeneticAlgorithm(BaseGeneticAlgorithm):
    """Simple implementation of genetic algorithm."""

    def __init__(
        self,
        state: GeneticAlgorithmState,
        eval_function: Callable,
        mutation_function: Callable,
        mating_function: Callable,
    ):
        self._state = state
        self._eval_function = eval_function
        self._mutation_function = mutation_function
        self._mating_function = mating_function

    def mutate(self, delta: float) -> None:
        # Note that this might break constraints
        # TODO: keep some of the best iterations
        new_pops = []
        for pop in self._state._population:
            new_pops.append(self._mutation_function(pop, delta=delta))
        self._state._population = np.array(new_pops)

    def select(self, keep_share: float) -> None:
        num_to_keep = int(self._state._population_size * keep_share)

        scores = []
        for pop in self._state._population:
            scores.append(self._eval_function(pop))

        # Note that this minimizes the scores
        order = np.argsort(scores)
        new_pops = self._state._population[order[:num_to_keep]]

        self._state._population = new_pops

    def mate(self) -> None:
        # Keep the best iterations then add mated ones
        init_size = len(self._state._population)
        pop_size = self._state._population_size
        new_pops = []
        while len(new_pops) < pop_size - init_size:
            ids = np.arange(init_size)
            candidate_ids = np.random.choice(ids, 2, replace=False)

            candidate0 = self._state._population[candidate_ids[0]]
            candidate1 = self._state._population[candidate_ids[1]]

            new_pops.append(self._mating_function([candidate0, candidate1], axis=0))

        self._state._population = np.append(self._state._population, np.array(new_pops), axis=0)

    def get_best(self) -> Sequence[float]:
        scores = []
        for pop in self._state._population:
            scores.append(self._eval_function(pop))

        min_id = np.argmin(scores)
        best_pop = self._state._population[min_id]
        return best_pop
