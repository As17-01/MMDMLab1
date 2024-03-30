
from typing import Optional, Sequence, List

import numpy as np

from src.mutations import reallocate_randomly

def mean_crossover(parents: Sequence[List[List[float]]], random_state: Optional[int] = None):
    np.random.seed(random_state)
    return np.mean(parents, axis=0).tolist()


def courier_2_parents_crossover(parents: Sequence[List[List[int]]], random_state: Optional[int] = None):
    np.random.seed(random_state)
    
    parent1_idx, parent2_idx = np.random.choice(len(parents), 2, replace=False)
    parent1, parent2 = parents[parent1_idx], parents[parent2_idx]

    child = parent1.copy()
    cross_courier_idx = np.random.randint(len(parent1))

    all_cities = {x for l in child for x in l}
    exchanged_cities = set(list(parent2[cross_courier_idx]))

    # Remove cities from array, which are exchanged
    for city in exchanged_cities:
        child = [path[path != city] for path in child]
    child[cross_courier_idx] = parent2[cross_courier_idx]

    # If a city is missing, add it to a random courier
    new_cities = {x for l in child for x in l}
    for missing_city in all_cities:
        if missing_city not in new_cities:
            child = reallocate_randomly(missing_city, child, random_state)

    return child
