from typing import List
import numpy as np


class DistanceCost:
    def __init__(self, salary, dist_matrix):
        self._salary = np.array(salary)
        self._dist_matrix = np.array(dist_matrix)
        
    def __call__(self, route: List[List[int]]) -> float:
        '''
        Функция принимает маршрут каждого курьера, 
        оплату каждого курьера за км (важно, что кол-во оплат = кол-во курьеров),
        матрицу дистанции между городами
        '''
        full_route = route.copy()

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
        