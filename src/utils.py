from typing import List


def create_eq_classes(demand: List[int]):
    result = {2: {}, 3: {}}

    # Cities, which can be replaced 1 to 1:
    for city1, demand1 in enumerate(demand):
        result[2][city1] = []
        for city2, demand2 in enumerate(demand):
            non_zeros = (demand1 != 0) and (demand2 != 0)
            not_equal = city1 != city2
            if (demand1 == demand2) and non_zeros and not_equal:
                result[2][city1].append(city2)

    # Cities, which can be replaced 2 to 1:
    for city1, demand1 in enumerate(demand):
        result[3][city1] = []
        for city2, demand2 in enumerate(demand):
            for city3, demand3 in enumerate(demand):
                non_zeros = (demand1 != 0) and (demand2 != 0) and (demand3 != 0)
                not_equal = (city1 != city2) and (city2 != city3) and (city1 != city3)
                if (demand1 == demand2 + demand3) and non_zeros and not_equal:
                    result[3][city1].append([city2, city3])

    return result


def find_city(city: int, distribution: List[List[List[int]]]):
    for courier_idx, route in enumerate(distribution):
        if city in route:
            city_idx = route.index(city)
            return courier_idx, city_idx
