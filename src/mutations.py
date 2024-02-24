from typing import Sequence

import numpy as np


def square_mutation(coordinates: Sequence[float], delta: float) -> Sequence[float]:
    new_coordinates = np.array(coordinates)

    distance = np.random.uniform(low=-delta, high=delta, size=len(coordinates))
    new_coordinates = new_coordinates + distance

    return new_coordinates
