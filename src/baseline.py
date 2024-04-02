import copy
from typing import Callable
from typing import List

import numpy as np

from src.base import BaseGeneticAlgorithm
from src.state import BaseGeneticAlgorithmState
from src.state import CouriersGeneticAlgorithmState
from src.utils import create_eq_classes


class BaselineGeneticAlgorithm(BaseGeneticAlgorithm):
    """Simple implementation of genetic algorithm."""

    def __init__(
        self,
        state: BaseGeneticAlgorithmState,
        eval_function: Callable,
        mutation_function: Callable,
        mating_function: Callable,
        random_state: int = 99,
    ):
        self._state = state
        self._eval_function = eval_function
        self._mutation_function = mutation_function
        self._mating_function = mating_function

        np.random.seed(random_state)

    def mutate(self, delta: float) -> None:
        # Note that this might break constraints for dots
        new_pops = []
        for pop in self._state.population:
            mutate_prob = np.random.uniform(0, 1)
            if mutate_prob <= 0.5:
                if isinstance(self._state, CouriersGeneticAlgorithmState):
                    eq_classes = create_eq_classes(self._state._demand)
                    distribution = self._mutation_function(pop, delta=delta, eq_classes=eq_classes)
                else:
                    distribution = self._mutation_function(pop, delta=delta)

                if distribution not in new_pops:
                    new_pops.append(distribution)
                else:
                    new_pops.append(pop)
            else:
                new_pops.append(pop)
        self._state.population = new_pops

    def select(self, keep_share: float) -> None:
        num_to_keep = int(self._state.population_size * keep_share)

        # Note that this minimizes the scores
        eval_results = []
        for pop in self._state.population:
            pop_eval_result = self._eval_function(pop)
            eval_results.append(pop_eval_result)

        new_population = []
        for _ in range(num_to_keep):
            new_pop_id = np.argmin(eval_results)
            new_population.append(self._state.population[new_pop_id])
            eval_results[new_pop_id] = 100000000

        self._state.population = new_population

    def mate(self) -> None:

        # Keep the best iterations then add mated ones
        init_size = len(self._state.population)
        pop_size = self._state.population_size

        new_pops = copy.deepcopy(self._state.population)
        while len(new_pops) < pop_size:
            candidate_ids = np.random.choice(np.arange(init_size), 2, replace=False)

            candidate0 = self._state.population[candidate_ids[0]]
            candidate1 = self._state.population[candidate_ids[1]]

            if isinstance(self._state, CouriersGeneticAlgorithmState):
                eq_classes = create_eq_classes(self._state._demand)
                distribution = self._mating_function([candidate0, candidate1], eq_classes=eq_classes)

                num_retries = 1
                while (not self._state._validate_capacity(distribution)) and (num_retries < 200):
                    distribution = self._mating_function([candidate0, candidate1], eq_classes=eq_classes)
                    num_retries += 1

                if (num_retries < 200) and (distribution not in new_pops):
                    new_pops.append(distribution)
            else:
                distribution = self._mating_function([candidate0, candidate1])
                new_pops.append(distribution)

        self._state.population = new_pops

    def get_best(self) -> List[List[float]]:
        eval_results = []
        for pop in self._state.population:
            pop_eval_result = self._eval_function(pop)
            eval_results.append(pop_eval_result)

        best_pop_id = np.argmin(eval_results)
        return self._state.population[best_pop_id]
