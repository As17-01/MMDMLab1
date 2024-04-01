import copy
from typing import List

import numpy as np


def mean_crossover(parents: List[List[float]]):
    return np.mean(parents, axis=0).tolist()


def reallocate_randomly(city: int, distribution: List[List[int]]) -> List[List[int]]:
    distribution = copy.deepcopy(distribution)

    num_couriers = len(distribution)
    _ = [path.remove(city) for path in distribution if city in path]

    recipient_idx = np.random.randint(num_couriers)
    recipient = distribution[recipient_idx]

    if len(recipient) > 0:
        place_idx = np.random.randint(len(recipient))
    else:
        place_idx = 0

    distribution[recipient_idx].insert(place_idx, city)
    return distribution


def courier_2_parents_crossover(parents: List[List[List[int]]], eq_classes) -> List[List[int]]:

    parent1_idx, parent2_idx = np.random.choice(len(parents), 2, replace=False)
    parent1, parent2 = parents[parent1_idx], parents[parent2_idx]

    child = copy.deepcopy(parent1)
    cross_courier_idx = np.random.randint(len(parent1))

    all_cities = {x for l in child for x in l}
    exchanged_cities = set(list(parent2[cross_courier_idx]))

    # Remove cities from array, which are exchanged
    for city in exchanged_cities:
        _ = [path.remove(city) for path in child if city in path]
    child[cross_courier_idx] = parent2[cross_courier_idx]

    # If a city is missing, add it to a random courier
    new_cities = {x for l in child for x in l}
    for missing_city in all_cities:
        if missing_city not in new_cities:
            child = reallocate_randomly(missing_city, child)

    return child
