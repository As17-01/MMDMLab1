from typing import Sequence

import numpy as np


def square_mutation(coordinates: Sequence[float], delta: float, random_state: int) -> Sequence[float]:
    np.random.seed(random_state)
    new_coordinates = np.array(coordinates)

    distance = np.random.uniform(low=-delta, high=delta, size=len(coordinates))
    new_coordinates = new_coordinates + distance

    return new_coordinates


def courier_mutation(distribution: Sequence[np.ndarray], delta: float, random_state: int):
    np.random.seed(random_state)
    num_couriers = len(distribution)

    num_jumps = int(np.random.exponential(delta))
    all_cities = {x for l in distribution for x in l}
    
    for _ in range(num_jumps):
        city = np.random.choice(all_cities, 1, replace=False)
        
        new_distribution = [ar.pop(city) for ar in new_distribution if city in ar]
        
        recipient_idx = np.random.randint(num_couriers)
        new_distribution[recipient_idx] = new_distribution[recipient_idx] + np.array(city)

    return new_distribution
