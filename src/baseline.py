import copy
from typing import Callable
from typing import Sequence

import numpy as np

from src.base import BaseGeneticAlgorithm
from src.state import BaseGeneticAlgorithmState
from src.state import CouriersGeneticAlgorithmState


class BaselineGeneticAlgorithm(BaseGeneticAlgorithm):
    """Simple implementation of genetic algorithm."""

    def __init__(
        self,
        state: BaseGeneticAlgorithmState,
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
        # Note that this might break constraints for dots
        new_pops = []
        for i, pop in enumerate(self._state.population):
            if i < self._state.population_size / 2:
                distribution = self._mutation_function(pop, delta=delta, random_state=self._random_state * 6 + i)

                if isinstance(self._state, CouriersGeneticAlgorithmState):
                    retry_factor = 1
                    while (not self._state._validate_capacity(distribution)) and (retry_factor < 100):
                        distribution = self._mutation_function(
                            pop, delta=delta, random_state=self._random_state * 6 + i + 100 * retry_factor
                        )
                        retry_factor += 1

                    if (retry_factor < 100) and (distribution not in new_pops):
                        new_pops.append(distribution)
                    else:
                        new_pops.append(pop)
                else:
                    new_pops.append(distribution)

            else:
                new_pops.append(pop)
        self._state.population = new_pops

    def select(self, keep_share: float) -> None:
        num_to_keep = int(self._state.population_size * keep_share)

        # Note that this minimizes the scores
        eval_results = []
        for pop in self._state.population:
            pop_eval_result = [f(pop) for f in self._eval_functions]
            eval_results.append(pop_eval_result)
        eval_results = np.array(eval_results)

        new_population = []
        for i in range(num_to_keep):
            np.random.seed(21 * self._random_state + i)
            weights = np.random.uniform(0, 1, size=len(self._eval_functions))

            new_pop_id = np.argmin(np.sum(eval_results * weights, axis=1))
            new_population.append(self._state.population[new_pop_id])
            eval_results[new_pop_id] = np.ones(len(self._eval_functions)) * 100000000

        self._state.population = new_population

    def mate(self) -> None:
        # Keep the best iterations then add mated ones
        init_size = len(self._state.population)
        pop_size = self._state.population_size

        new_pops = copy.deepcopy(self._state.population)
        i = 0
        while len(new_pops) < pop_size:
            i += 1

            np.random.seed(8 * self._random_state + i * 4)
            candidate_ids = np.random.choice(np.arange(init_size), 2, replace=False)

            candidate0 = self._state.population[candidate_ids[0]]
            candidate1 = self._state.population[candidate_ids[1]]

            if candidate0 == candidate1:
                continue

            distribution = self._mating_function([candidate0, candidate1], random_state=self._random_state + i * 9)

            if isinstance(self._state, CouriersGeneticAlgorithmState):
                retry_factor = 1
                while (not self._state._validate_capacity(distribution)) and (retry_factor < 100):
                    distribution = self._mating_function(
                        [candidate0, candidate1], random_state=self._random_state + i * 9 + 77 * retry_factor
                    )
                    retry_factor += 1

                if (retry_factor < 100) and (distribution not in new_pops):
                    new_pops.append(distribution)
            else:
                new_pops.append(distribution)

        self._state.population = new_pops

    def get_best(self) -> Sequence[Sequence[float]]:
        eval_results = []
        for pop in self._state.population:
            pop_eval_result = [f(pop) for f in self._eval_functions]
            eval_results.append(pop_eval_result)
        eval_results = np.array(eval_results)

        scores = []
        for pop_eval in eval_results:
            ordering_score = np.sum(np.all(eval_results < pop_eval, axis=1))
            scores.append(ordering_score)

        min_ids = np.arange(len(scores))[np.array(scores) == 0]
        best_pops = [self._state.population[i] for i in min_ids]

        # In case of equality
        best_pops_unique = []
        for pop in best_pops:
            if pop not in best_pops_unique:
                best_pops_unique.append(pop)
        return best_pops_unique
