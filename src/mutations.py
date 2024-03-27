from typing import Sequence

import numpy as np


def square_mutation(coordinates: Sequence[float], delta: float) -> Sequence[float]:
    new_coordinates = np.array(coordinates)

    distance = np.random.uniform(low=-delta, high=delta, size=len(coordinates))
    new_coordinates = new_coordinates + distance

    return new_coordinates

def coruier_mutation(courier, num_points_to_move=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    num_couriers = len(courier)
    
    for _ in range(num_points_to_move):
        donor_idx = np.random.choice([i for i, points in enumerate(courier) if len(points) > 1])
        point_idx = np.random.randint(len(courier[donor_idx]))
        point_to_move = courier[donor_idx].pop(point_idx)
        
        recipient_idxs = list(range(num_couriers))
        recipient_idxs.remove(donor_idx)
        recipient_idx = np.random.choice(recipient_idxs)
        
        courier[recipient_idx].append(point_to_move)

    return courier
