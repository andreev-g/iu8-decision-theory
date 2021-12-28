import numpy as np
import pandas as pd
from tabulate import tabulate

from src.utility import print_separator


class SimplexMethod:
    def __init__(self, a, b, c, mode):
        self.A = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.mode = mode

        matrix = np.c_[b, a]
        self.matrix = np.r_[matrix, [[0, *self.c]]]
        self.origin_matrix = np.copy(self.matrix)
        # берем функционал с обратным знаком
        self.matrix[-1] *= -1

        s0 = 'S0'
        self.columns = [s0] + [f'x_{i + 1}' for i in range(len(self.A[0]))]
        f = 'F'
        self.index = [f'x_{i + 1 + len(self.A[0])}' for i in range(len(self.A[:, 0]))] + [f]
        self.answer = []

    def find_pivot(self):
        """
        Функция поиска разрешающего элемента (при поиске оптимального решения)
        :return: координаты разрешающего элемента
        """
        j = 0
        if str(self.mode) == 'min':
            j = np.argmax(self.matrix[-1:][0][1:]) + 1
        elif str(self.mode) == 'max':
            j = np.argmin(self.matrix[-1:][0][1:]) + 1

        permissive_column = self.matrix[:, j]
        free_member_column = self.matrix[:, 0]
        with np.errstate(divide='ignore'):
            np.seterr(invalid='ignore')
            arr = free_member_column[:-1] / permissive_column[:-1]
            arr[arr < 0] = np.inf
            arr[~np.isfinite(arr)] = np.inf
            i = arr.argmin()
            assert arr[i] != np.inf, f'Система имеет бесконечно много решений'
        return i, j

    def find_base_pivot(self):
        """
        Функция поиска разрешающего элмента (при поиске опронго решения)
        :return: False - поиск опороного решения не требуется, координаты разрешающего элемента
        """
        first_column = self.matrix[:-1, 0].flatten()
        negative_row = np.where(first_column < 0, first_column, np.inf).argmin()
        if negative_row == 0 and first_column[negative_row] > 0:
            return False, 0, 0
        row = self.matrix[negative_row][1:]
        assert np.any(row < 0), f'Система не имеет решений'
        j = np.where(row < 0, row, np.inf).argmin()
        j += 1
        base_column = self.matrix[:-1, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            arr = first_column / base_column
            arr[arr <= 0] = np.inf
            i = arr.argmin()
            assert arr[i] != np.inf, f'Система имеет бесконечное число решений'
        return True, i, j

    def swap(self, pivot):
        """
        Функция замены базиса
        :param pivot: индекс разрешающего элемента
        :return: таблица с новым базисом
        """
        self.index[pivot[0]], self.columns[pivot[1]] = self.columns[pivot[1]], self.index[pivot[0]]
        pivot_value = self.matrix[pivot]
        matrix_size = self.matrix.shape
        new_matrix = np.zeros((matrix_size[0], matrix_size[1]))

        assert pivot_value != 0, "Симплекс таблица не имеет решений"

        # элементы симплкс таблицы, кроме разрещей строки и разрешающего столбца
        for i, j in [(i, j) for i in range(matrix_size[0]) for j in range(matrix_size[1])]:
            if i == pivot[0] or j == pivot[1]:
                continue
            new_matrix[i, j] = self.matrix[i, j] - self.matrix[pivot[0], j] * self.matrix[i, pivot[1]] / pivot_value

        # разрешающий столбец
        for i in range(matrix_size[0]):
            if i == pivot[0]:
                continue
            new_matrix[i, pivot[1]] = -self.matrix[i, pivot[1]] / pivot_value

        # разрешающая строка
        for j in range(matrix_size[1]):
            if j == pivot[1]:
                continue
            new_matrix[pivot[0], j] = self.matrix[pivot[0], j] / pivot_value

        new_matrix[pivot] = 1 / self.matrix[pivot]

        return new_matrix

    def find_base_solution(self):
        """
        Функция поиска опорного решения
        """
        while True:
            solution, i, j = self.find_base_pivot()
            if not solution:
                break
            print_separator()
            pivot = (i, j)
            print('Индекс разрешающего элемента: ', pivot)
            self.matrix = self.swap(pivot)
            self.print()

    def find_optimal_solution(self):
        """
        Функция поиска оптимального решения
        :return:
        """
        print_separator()
        print('Ищем оптимальное решение:')
        if str(self.mode) == 'min':
            while not all(i < 0 for i in self.matrix[-1][1:]):
                print_separator()
                pivot = self.find_pivot()
                print('Индекс разрешающего элемента: ', pivot)
                self.matrix = self.swap(pivot)
                self.print()
        if str(self.mode) == 'max':
            while not all(i > 0 for i in self.matrix[-1][1:]):
                print_separator()
                pivot = self.find_pivot()
                print('Индекс разрешающего элемента: ', pivot)
                self.matrix = self.swap(pivot)
                self.print()
        print_separator()

    def check_solution(self, solution):
        """
        Функция проверки решения
        :param solution: решение
        :return:
        """
        f = 0
        for idx, item in enumerate(solution):
            if idx == 0:
                continue
            f += self.origin_matrix[-1][idx] * item
        assert solution[0] == f, f'Результат оптимального решения не совпадает с коэффициентами F={solution[0]}, f={f}'

        for r, row in enumerate(self.origin_matrix[:-1]):
            multiplication = row[1:] * solution[1:]
            limit = np.sum(multiplication)
            assert row[0] >= limit, f'Ограничение #{r} нарушено: {limit} <= {row[0]}'
            print(f'Ограничение #{r}: {limit} <= {row[0]}')

        print('Решение верно!')

    def get_result(self):
        self.print()
        self.find_base_solution()
        self.find_optimal_solution()
        for i, _ in enumerate(self.index):
            if f'x_{i + 1}' in self.index:
                self.answer.append(self.matrix[self.index.index(f'x_{i + 1}')][0])
            else:
                self.answer.append(0)

        print('x1 =', self.answer[0])
        print('x2 =', self.answer[1])
        print('x3 =', self.answer[2])
        print('F =', self.matrix[-1, 0])
        self.answer.insert(0, self.matrix[-1, 0])

        return self.answer

    def print(self):
        df = pd.DataFrame(data=self.matrix, index=self.index, columns=self.columns)
        print(
            tabulate(
                df.applymap(lambda x: x if x != 0 else 0.),
                headers="keys",
                tablefmt="psql"
            )
        )
