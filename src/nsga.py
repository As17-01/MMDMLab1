import copy
from typing import Callable
from typing import List

import numpy as np

from src.base import BaseGeneticAlgorithm
from src.baseline import BaselineGeneticAlgorithm
from src.state import BaseGeneticAlgorithmState
from src.utils import create_eq_classes


class NSGeneticAlgorithm(BaselineGeneticAlgorithm):

    def __init__(
        self,
        state: BaseGeneticAlgorithmState,
        eval_functions: List[Callable],
        mutation_function: Callable,
        mating_function: Callable,
        random_state: int = 99,
    ):
        self._state = state
        self._eval_functions = eval_functions
        self._mutation_function = mutation_function
        self._mating_function = mating_function

        np.random.seed(random_state)

    def get_best(self) -> List[List[float]]:
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
            if np.round(pop, 8).tolist() not in np.round(best_pops_unique, 8).tolist():
                best_pops_unique.append(pop)
        return best_pops_unique


    def select(self, keep_share):
        fronts = self.fast_non_dominated_sort()
        
        crowding_distances = self.calculate_crowding_distance(fronts)
        
        keep_num = int(len(self._state.population) * keep_share)
        
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > keep_num:
                sorted_front = sorted(front, key=lambda x: crowding_distances[x], reverse=True)
                new_population.extend(sorted_front[:keep_num - len(new_population)])
                break
            new_population.extend(front)
        
        self._state.population = [self._state.population[i] for i in new_population]

    def calculate_crowding_distance(self, fronts):
        crowding_distances = {index: 0 for index in range(len(self._state.population))}
        
        for front in fronts:
            if len(front) == 0:
                continue

            for func_index in range(len(self._eval_functions)):
                scores = [(self._eval_functions[func_index](self._state.population[index]), index) for index in front]
                scores.sort()

                crowding_distances[scores[0][1]] = float('inf')
                crowding_distances[scores[-1][1]] = float('inf')

                score_range = scores[-1][0] - scores[0][0]
                if score_range == 0:
                    continue

                for j in range(1, len(scores) - 1):
                    crowding_distances[scores[j][1]] += (scores[j + 1][0] - scores[j - 1][0]) / score_range

        return crowding_distances

    def fast_non_dominated_sort(self):
        population = self._state.population
        fronts = [[]]
        
        domination_counts = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self.dominates(i, j):
                        dominated_solutions[i].append(j)
                    elif self.dominates(j, i):
                        domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            fronts.append(next_front)
            current_front += 1
        
        return fronts[:-1]

    def dominates(self, index1, index2):
        better_in_one = False
        scores1 = [f(self._state.population[index1]) for f in self._eval_functions]
        scores2 = [f(self._state.population[index2]) for f in self._eval_functions]
        
        for a, b in zip(scores1, scores2):
            if a > b: 
                return False
            elif a < b:
                better_in_one = True
                
        return better_in_one