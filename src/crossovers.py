
from typing import Optional, List

import numpy as np

def crossover(parents: List[List[List[int]]], random_state: Optional[int] = None) -> List[List[int]]:
    np.random.seed(random_state)
    
    parent1_index, parent2_index = np.random.choice(len(parents), 2, replace=False)
    parent1, parent2 = parents[parent1_index], parents[parent2_index]
    
    courier_index = np.random.randint(len(parent1))

    child = [list(courier) for courier in parent1]
    
    child[courier_index] = list(parent2[courier_index])
    
    all_points = set(range(1, max(max(courier) for courier in parent1) + 1))

    new_courier_points = set(child[courier_index])
    for i, courier in enumerate(child):
        if i != courier_index:
            courier_points = set(courier)
            courier[:] = list(courier_points - new_courier_points)
            new_courier_points |= courier_points

    missing_points = all_points - new_courier_points
    for point in missing_points:
        min_len_courier_index = min((len(courier), i) for i, courier in enumerate(child) if i != courier_index)[1]
        child[min_len_courier_index].append(point)

    for courier in child:
        if len(courier) == 0:
            idx = np.random.randint(0, len(child[courier_index]))
            courier.append(child[courier_index].pop(idx))
    
    return child