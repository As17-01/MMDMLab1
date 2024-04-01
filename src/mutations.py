import copy
from typing import List

import numpy as np

from src.utils import find_city


def square_mutation(coordinates: List[float], delta: float) -> List[float]:
    new_coordinates = np.array(coordinates)

    distance = np.random.uniform(low=-delta, high=delta, size=len(coordinates))
    new_coordinates = new_coordinates + distance

    return new_coordinates.tolist()


def courier_mutation(distribution: List[List[int]], delta: float, eq_classes):
    distribution = copy.deepcopy(distribution)

    num_jumps = np.random.randint(int(delta))
    all_cities = {x for l in distribution for x in l}

    city = np.random.randint(1, len(list(all_cities)) + 1)

    if num_jumps == 0:
        pass

    # Sorting
    elif num_jumps == 1:
        city_address = find_city(city, distribution)
        distribution[city_address[0]].remove(city)

        if len(distribution[city_address[0]]) > 0:
            place_idx = np.random.randint(len(distribution[city_address[0]]))
        else:
            place_idx = 0

        distribution[city_address[0]].insert(place_idx, city)

    # Exchange 1 to 1
    elif num_jumps == 2:
        if len(eq_classes[2][city]) > 0:
            replacement_idx = np.random.randint(len(eq_classes[2][city]))
            replacement_city = eq_classes[2][city][replacement_idx]

            city_address = find_city(city, distribution)
            rep_city_address = find_city(replacement_city, distribution)

            distribution[city_address[0]].remove(city)
            distribution[rep_city_address[0]].remove(replacement_city)

            distribution[city_address[0]].insert(city_address[1], replacement_city)
            distribution[rep_city_address[0]].insert(rep_city_address[1], city)

    # Exchange 2 to 1
    elif num_jumps >= 3:
        if len(eq_classes[3][city]) > 0:
            replacement_idx = np.random.randint(len(eq_classes[3][city]))
            replacement_cities = eq_classes[3][city][replacement_idx]

            city_address = find_city(city, distribution)

            new_city_addresses = [find_city(new_city, distribution)[0] for new_city in replacement_cities]
            if len(np.unique(new_city_addresses)) == 1:

                distribution[city_address[0]].remove(city)
                for new_city in replacement_cities:
                    distribution[new_city_addresses[0]].remove(new_city)

                distribution[new_city_addresses[0]].insert(0, city)
                for new_city in replacement_cities:
                    distribution[city_address[0]].insert(0, new_city)

    return distribution
