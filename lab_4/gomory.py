import numpy as np

from lab_4.simplex import Simplex

from src.utility import print_separator


class Gomory:
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        self.a = np.column_stack((self.a, np.eye(self.b.size)))
        self.c = np.append(self.c, np.zeros(self.b.size))

        self.simplex = Simplex(self.a, self.b, self.c, "max")

    @staticmethod
    def check_solution(solution):
        for el in solution:
            if not el.is_integer():
                return False
        return True

    @staticmethod
    def find_max_fractional_part(solution):
        solution = np.modf(solution)
        return np.argmax(solution[0])

    def solution(self):
        integer_solution = False
        while not integer_solution:
            self.simplex.get_result()
            integer_solution = self.check_solution(self.simplex.answer)
            if integer_solution:
                break
            idx = self.find_max_fractional_part(self.simplex.answer[:-1])
            new_limit_array = np.modf(self.simplex.matrix[idx][1:])[0]
            for index, el in enumerate(new_limit_array):
                if el < 0:
                    new_limit_array[index] = el + 1
            new_limit_array *= -1
            new_limit_array[new_limit_array == -0.0] = 0.0
            new_a = np.vstack([self.simplex.A, new_limit_array])
            new_limit = np.modf(self.simplex.answer[idx])[0]
            new_b = np.append(self.simplex.b, new_limit * -1)
            added_column = np.zeros(new_b.size)
            added_column[-1] = 1
            new_a = np.column_stack((new_a, added_column))
            new_c = np.append(self.simplex.c, 0)
            self.simplex = Simplex(new_a, new_b, new_c, "max")
            print_separator()

        return self.simplex.answer
