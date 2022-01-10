import numpy as np

from lab_5.simplex import Simplex


class PlayersGame:
    def __init__(self, strategy_matrix):
        self.strategy_matrix = np.array(strategy_matrix)
        self.result_a = []
        self.result_b = []

        self.simplex_table_a = None
        self.simplex_table_b = None
        self.b_a = None
        self.b_b = None
        self.c_a = None
        self.c_b = None

    def preparation_before_simplex(self):
        self.simplex_table_a = np.transpose(self.strategy_matrix) * (-1)
        size_b, size_c = self.simplex_table_a.shape
        self.b_a = np.ones(size_b) * (-1)
        self.c_a = np.ones(size_c)
        self.simplex_table_b = self.strategy_matrix
        size_b, size_c = self.simplex_table_b.shape
        self.b_b = np.ones(size_b)
        self.c_b = np.ones(size_c)

    @staticmethod
    def preparation_after_simplex(result, size):
        g = 1/result[0]
        result[0] = g
        for idx, el in enumerate(result):
            if idx == 0:
                continue
            result[idx] = el * g
        return result[:size + 1]

    def simplex_method(self):
        print("Найдем оптимальную стратегию игрока А")
        player_a = Simplex(self.simplex_table_a, self.b_a, self.c_a, "min")
        try:
            self.result_a = player_a.get_result()
        except AssertionError:
            print("Для игрока А решения нет!")
        self.result_a = self.preparation_after_simplex(self.result_a, self.c_a.size)

        print("\nНайдем оптимальную стратегию игрока В\n")
        player_b = Simplex(self.simplex_table_b, self.b_b, self.c_b, "max")
        try:
            self.result_b = player_b.get_result()
        except AssertionError:
            print("Для игрока В решения нет!")
        self.result_b = self.preparation_after_simplex(self.result_b, self.c_b.size)

    def solve(self):
        self.preparation_before_simplex()
        self.simplex_method()
        self.result_a = [round(x, 3) for x in self.result_a]
        self.result_b = [round(x, 3) for x in self.result_b]
        self.check_solution()
        return self.result_a, self.result_b

    def print_solution(self):
        print("Оптимальная стратегия игрока А:")
        print(f"g = {self.result_a[0]}")
        for idx, el in enumerate(self.result_a[1:]):
            print(f"x{idx + 1} = {el}")
        print("\nОптимальная стратегия игрока B:")
        print(f"g = {self.result_b[0]}")
        for idx, el in enumerate(self.result_b[1:]):
            print(f"y{idx + 1} = {el}")

    def check_solution(self):
        assert np.sum(self.result_a[1:]) == 1.0, f"Решение игрока А не верно"
        assert np.sum(self.result_b[1:]) == 1.0, f"Решение игрока B не верно"
