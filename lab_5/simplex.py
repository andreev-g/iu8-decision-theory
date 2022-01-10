import numpy as np
import pandas as pd
from tabulate import tabulate

from src.utility import print_separator


class Simplex:
    def __init__(self, a, b, c, mode):
        self.A = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.mode = mode

        matrix = np.c_[b, a]
        self.matrix = np.r_[matrix, [[0, *c]]]
        self.origin_matrix = np.copy(self.matrix)
        # берем функционал с обратным знаком
        self.matrix[-1] *= -1

        s0 = "S0"
        self.columns = [s0] + [f'x{i + 1}' for i in range(len(self.A[0]))]
        f = "F"
        self.index = [f'x{i + 1 + len(self.A[0])}' for i in range(len(self.A[:, 0]))] + [f]
        self.answer = []

    def find_pivot(self):
        j = 0
        if self.mode == "min":
            j = np.argmax(self.matrix[-1:][0][1:]) + 1
        elif self.mode == "max":
            j = np.argmin(self.matrix[-1:][0][1:]) + 1

        permissive_column = self.matrix[:, j]
        free_member_column = self.matrix[:, 0]
        with np.errstate(divide='ignore'):
            np.seterr(invalid='ignore')
            arr = free_member_column[:-1] / permissive_column[:-1]
            arr[arr < 0] = np.inf
            arr[~np.isfinite(arr)] = np.inf
            i = arr.argmin()
            assert arr[i] != np.inf, f"Система имеет бесконечно много решений"
        return i, j

    def find_base_pivot(self):
        first_column = self.matrix[:-1, 0].flatten()
        negative_row_idx = 0
        for idx, el in enumerate(first_column):
            if el < 0:
                negative_row_idx = idx
                break
        if negative_row_idx == 0 and first_column[negative_row_idx] > 0:
            return False, 0, 0
        row = self.matrix[negative_row_idx][1:]
        assert np.any(row < 0), f"Система не имеет решений"
        j = np.where(row < 0, row, np.inf).argmin()
        j += 1
        base_column = self.matrix[:-1, j]
        with np.errstate(divide='ignore'):
            arr = first_column / base_column
            arr[arr <= 0] = np.inf
            i = arr.argmin()
            assert arr[i] != np.inf, f"Система имеет бесконечно много решений"
        return True, i, j

    def swap(self, pivot):
        self.index[pivot[0]], self.columns[pivot[1]] = self.columns[pivot[1]], self.index[pivot[0]]
        pivot_value = self.matrix[pivot]
        matrix_size = self.matrix.shape
        new_matrix = np.zeros((matrix_size[0], matrix_size[1]))

        new_matrix[pivot] = 1 / self.matrix[pivot]

        # разрешающая строка
        for j in range(matrix_size[1]):
            if j == pivot[1]:
                continue
            new_matrix[pivot[0], j] = self.matrix[pivot[0], j] / pivot_value

        # разрешающий столбец
        for i in range(matrix_size[0]):
            if i == pivot[0]:
                continue
            new_matrix[i, pivot[1]] = -self.matrix[i, pivot[1]] / pivot_value

        # остальные элементы симплкс таблицы, кроме разрещей строки и разрешающего столбца
        for i, j in [(i, j) for i in range(matrix_size[0]) for j in range(matrix_size[1])]:
            if i == pivot[0] or j == pivot[1]:
                continue
            new_matrix[i, j] = self.matrix[i, j] - self.matrix[pivot[0], j] * self.matrix[i, pivot[1]] / pivot_value

        return new_matrix

    def find_base_solution(self):
        flag = True
        while True:
            solution, i, j = self.find_base_pivot()
            if not solution:
                break
            if flag:
                print("ПОИСК ОПОРНОГО РЕШЕНИЯ")
                flag = False
            print_separator()
            pivot = (i, j)
            print("Индекс разрешающего элемента: ", pivot)
            self.matrix = self.swap(pivot)
            self.print()
        return True

    def find_optimal_solution(self):
        flag = True
        if self.mode == "min":
            while not all(i < 0 for i in self.matrix[-1][1:]):
                if flag:
                    print("Поиск оптимального решения")
                    flag = False
                print_separator()
                pivot = self.find_pivot()
                print("Индекс разрешающего элемента: ", pivot)
                self.matrix = self.swap(pivot)
                self.print()
        if self.mode == "max":
            while not all(i > 0 for i in self.matrix[-1][1:]):
                if flag:
                    print("Поиск оптимального решения")
                    flag = False
                print_separator()
                pivot = self.find_pivot()
                print("Индекс разрешающего элемента: ", pivot)
                self.matrix = self.swap(pivot)
                self.print()
        print_separator()

    def check_solution(self, solution):
        f = 0
        for idx, item in enumerate(solution):
            if idx == 0:
                continue
            f += self.origin_matrix[-1][idx] * item
        assert solution[0] == f, f"Результат оптимального решения не совпадает с коэффициентами F={solution[0]}, f={f}"

        for r, row in enumerate(self.origin_matrix[:-1]):
            multiplication = row[1:] * solution[1:]
            limit = np.sum(multiplication)
            assert row[0] >= limit, f"Ограничение №{r} нарушено: {limit} <= {row[0]}"
            print(f"Ограничение №{r}: {limit} <= {row[0]}")

        print("Решение верно!")

    def get_result(self):
        self.print()
        self.find_base_solution()
        self.find_optimal_solution()
        for i in range(self.b.size + self.c.size):
            if f"x{i + 1}" in self.index:
                self.answer.append(self.matrix[self.index.index(f"x{i + 1}")][0])
            else:
                self.answer.append(0)

        self.answer.insert(0, self.matrix[-1, 0])

        return self.answer

    def print(self):
        df = pd.DataFrame(data=np.round(self.matrix, 3), index=self.index, columns=self.columns)
        print(
            tabulate(
                df.applymap(lambda x: x if x != 0 else 0.),
                headers="keys",
                tablefmt="psql"
            )
        )
