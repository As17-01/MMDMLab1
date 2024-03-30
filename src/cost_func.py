from typing import List
import numpy as np


class DistanceCost:
    def __init__(self, salary, dist_matrix, demand, capacity):
        self._salary = np.array(salary)
        self._dist_matrix = np.array(dist_matrix)
        self._demand = demand
        self._capacity = capacity

    def _validate_capacity(self, courier_id: int, courier_route: List[int]):
        if np.sum(np.array(self._demand)[np.array(courier_route)]) > self._capacity[courier_id]:
            return False
        return True
        
    def __call__(self, route: List[List[int]]) -> float:
        '''
        Функция принимает маршрут каждого курьера, 
        оплату каждого курьера за км (важно, что кол-во оплат = кол-во курьеров),
        матрицу дистанции между городами
        '''
        full_route = route.copy()

        for i, courier_route in enumerate(full_route):
            if not self._validate_capacity(i, courier_route):
                return 100000000

        # Добавляем 0 в начало и конец пути каждого курьера, тк они стартуют и возвращаются в депот
        for i in full_route:
            i.insert(0,0)
            i.append(0)
            
        total_distance = [] # общие косты всех курьеров   
        for courier in full_route:
            cities_num = len(courier)

            courier_distance = [] # косты 1 курьера за весь путь 
            for i in range(cities_num-1):
                start = courier[i]
                end = courier[i+1]
                distance = self._dist_matrix[start,end]
                courier_distance.append(distance)
                
            total_distance.append(sum(courier_distance))
            
        total_distance = np.array(total_distance)
        total_costs = np.multiply(total_distance, self._salary).sum() 
        
        return total_costs
        