from typing import Callable, Sequence

import numpy as np

from src.base import BaseGeneticAlgorithm
from src.state import Any


class BaselineGeneticAlgorithm(BaseGeneticAlgorithm):
    """Simple implementation of genetic algorithm."""

    def __init__(
        self,
        state: Any,
        eval_functions: Sequence[Callable],
        mutation_function: Callable,
        mating_function: Callable,
        random_state: int = 99,
    ):
        self._state = state
        self._eval_functions = eval_functions
        self._mutation_function = mutation_function
        self._mating_function = mating_function
        self._random_state = random_state

    def mutate(self, delta: float) -> None:
        # Note that this might break constraints
        # TODO: keep some of the best iterations
        new_pops = []
        for pop in self._state._population:
            new_pops.append(self._mutation_function(pop, delta=delta))
        self._state._population = np.array(new_pops)

    def select(self, keep_share: float) -> None:
        num_to_keep = int(self._state._population_size * keep_share)
        np.random.seed(self._random_state + int(np.max(self._state._population) - np.min(self._state._population) * 21))

        # Note that this minimizes the scores
        eval_results = []
        for pop in self._state._population:
            pop_eval_result = [f(pop) for f in self._eval_functions]
            eval_results.append(pop_eval_result)
        eval_results = np.array(eval_results)

        new_population = []
        while len(new_population) < num_to_keep:
            weights = np.random.uniform(0, 1, size=len(self._eval_functions))

            # TODO: Delete andom comment
            new_pop_id = np.argmin(np.sum(eval_results * weights, axis=1))
            new_population.append(self._state._population[new_pop_id])
            eval_results[new_pop_id] = np.array([np.inf] * len(self._eval_functions))

        self._state._population = np.array(new_population)

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

        new_pops_array = np.array(new_pops)
        self._state._population = np.append(
            self._state._population, new_pops_array, axis=0
        )

    def get_best(self) -> Sequence[Sequence[float]]:
        eval_results = []
        for pop in self._state._population:
            pop_eval_result = [f(pop) for f in self._eval_functions]
            eval_results.append(pop_eval_result)
        eval_results = np.array(eval_results)

        scores = []
        for pop_eval in eval_results:
            ordering_score = np.sum(np.all(eval_results < pop_eval, axis=1))
            scores.append(ordering_score)

        min_ids = np.array(scores) == 0
        best_pops = self._state._population[min_ids]
        return best_pops
