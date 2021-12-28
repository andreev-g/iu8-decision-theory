import math
import itertools
import numpy as np


class BruteForce:
    
    def __init__(self, a, b, c, optimal_value) -> None:
        self._a = np.array(a)
        self._b = np.array(b)
        self._c = np.array(c)
        self._optimal_value = optimal_value

    def brute_optimal(self):

        results = {}
        max_x = math.ceil(self._optimal_value / self._c[self._c > 0].min())

        for combination in itertools.product(np.arange(max_x), repeat=self._c.size):
            number_of_valid_constraints = 0
            for i in range(self._b.size):
                constraints = self._a[i] * combination
                if np.sum(constraints) <= self._b[i]:
                    number_of_valid_constraints += 1

            if number_of_valid_constraints == self._b.size:
                result = np.sum(combination * self._c)
                results[result] = combination
                print(f"F{combination}", "=", result)

        return max(results.keys()), results[max(results.keys())]
