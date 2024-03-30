
from typing import Optional, Sequence, List

import numpy as np


def mean_crossover(parents: Sequence[List[List[int]]], random_state: Optional[int] = None):
    np.random.seed(random_state)
    return np.mean(parents, axis=0)


def courier_2_parents_crossover(parents: Sequence[List[List[int]]], random_state: Optional[int] = None):
    np.random.seed(random_state)
    
    parent1_idx, parent2_idx = np.random.choice(len(parents), 2, replace=False)
    parent1, parent2 = parents[parent1_idx], parents[parent2_idx]

    child = parent1.copy()
    cross_courier_idx = np.random.randint(len(parent1))

    all_cities = {x for l in child for x in l}
    exhanged_cities = set(*parent2[cross_courier_idx])

    # Remove cities from array, which are exchanged
    for exch_city in exhanged_cities:
        child = [ar.pop(exch_city) for ar in child if exch_city in ar]
    child[cross_courier_idx] = parent2[cross_courier_idx]

    # If a city is missing, add it to a random courier
    new_cities = {x for l in child for x in l}
    for missing_city in all_cities:
        if missing_city not in new_cities:
            idx = np.random.randint(len(child))
            child[idx] = child[idx] + np.array(missing_city)

    return child
