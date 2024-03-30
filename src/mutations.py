from typing import Sequence

import numpy as np
import copy


def square_mutation(coordinates: Sequence[float], delta: float, random_state: int) -> Sequence[float]:
    np.random.seed(random_state)
    new_coordinates = np.array(coordinates)

    distance = np.random.uniform(low=-delta, high=delta, size=len(coordinates))
    new_coordinates = new_coordinates + distance

    return new_coordinates.tolist()

def reallocate_randomly(city: int, distribution, random_state: int):
    distribution = copy.deepcopy(distribution)
    np.random.seed(random_state)

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

def courier_mutation(distribution: Sequence[np.ndarray], delta: float, random_state: int):
    np.random.seed(random_state)

    num_jumps = int(np.random.exponential(delta))
    all_cities = {x for l in distribution for x in l}
    
    for i in range(num_jumps):
        city = np.random.randint(1, len(list(all_cities)) + 1)
        distribution = reallocate_randomly(city, distribution, random_state + 100 * i)

    return distribution
