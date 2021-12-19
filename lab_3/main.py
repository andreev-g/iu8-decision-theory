import sys
import yaml
import math
import numpy
import pandas
import pydantic
import itertools
import typing as t


class Problem(pydantic.BaseModel):
    c: t.List[float]
    A: t.List[t.List[float]]
    b: t.List[float]

    @classmethod
    def from_yaml(cls, filename: str) -> "Problem":
        with open(filename, "r") as f:
            data = yaml.load(f, yaml.CLoader)
            p = cls(**data)
            return p


class MinTowarding:
    def __str__(self):
        return 'min'

class MaxTowarding:
    def __str__(self):
        return 'max'

def print_separator():
    print('------------------------------------')

class SimplexMethod:
    def __init__(self, a, b, c, mode):
        self.A = numpy.array(a)
        self.b = numpy.array(b)
        self.c = numpy.array(c)
        self.mode = mode

        matrix = numpy.c_[b, a]
        self.matrix = numpy.r_[matrix, [[0, *c]]]
        self.origin_matrix = numpy.copy(self.matrix)
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
            j = numpy.argmax(self.matrix[-1:][0][1:]) + 1
        elif str(self.mode) == 'max':
            j = numpy.argmin(self.matrix[-1:][0][1:]) + 1

        permissive_column = self.matrix[:, j]
        free_member_column = self.matrix[:, 0]
        with numpy.errstate(divide='ignore'):
            numpy.seterr(invalid='ignore')
            arr = free_member_column[:-1] / permissive_column[:-1]
            arr[arr < 0] = numpy.inf
            arr[~numpy.isfinite(arr)] = numpy.inf
            i = arr.argmin()
            assert arr[i] != numpy.inf, f'Система имеет бесконечно много решений'
        return i, j

    def find_base_pivot(self):
        """
        Функция поиска разрешающего элмента (при поиске опронго решения)
        :return: False - поиск опороного решения не требуется, координаты разрешающего элемента
        """
        first_column = self.matrix[:-1, 0].flatten()
        negative_row = numpy.where(first_column < 0, first_column, numpy.inf).argmin()
        if negative_row == 0 and first_column[negative_row] > 0:
            return False, 0, 0
        row = self.matrix[negative_row][1:]
        assert numpy.any(row < 0), f'Система не имеет решений'
        j = numpy.where(row < 0, row, numpy.inf).argmin()
        j += 1
        base_column = self.matrix[:-1, j]
        with numpy.errstate(divide='ignore'):
            arr = first_column / base_column
            arr[arr <= 0] = numpy.inf
            i = arr.argmin()
            assert arr[i] != numpy.inf, f'Система имеет бесконечное число решений'
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
        new_matrix = numpy.zeros((matrix_size[0], matrix_size[1]))

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
            print(pandas.DataFrame(data=self.matrix, index=self.index, columns=self.columns))

    def find_optimal_solution(self):
        """
        Функция поиска оптимального решения
        :return:
        """
        print('ПОИСК ОПТИМАЛЬНОГО РЕШЕНИЯ')
        if str(self.mode) == 'min':
            while not all(i < 0 for i in self.matrix[-1][1:]):
                print_separator()
                pivot = self.find_pivot()
                print('Индекс разрешающего элемента: ', pivot)
                self.matrix = self.swap(pivot)
                print(pandas.DataFrame(data=self.matrix, index=self.index, columns=self.columns))
        if str(self.mode) == 'max':
            while not all(i > 0 for i in self.matrix[-1][1:]):
                print_separator()
                pivot = self.find_pivot()
                print('Индекс разрешающего элемента: ', pivot)
                self.matrix = self.swap(pivot)
                print(pandas.DataFrame(data=self.matrix, index=self.index, columns=self.columns))
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
            limit = numpy.sum(multiplication)
            assert row[0] >= limit, f'Ограничение №{r} нарушено: {limit} <= {row[0]}'
            print(f'Ограничение №{r}: {limit} <= {row[0]}')

        print('Решение верно!')

    def get_result(self):
        print(pandas.DataFrame(data=self.matrix, index=self.index, columns=self.columns))
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

class TreeNode:
    def __init__(self, f, value: SimplexMethod):
        self.left = None
        self.right = None
        self.f = f
        self.value = value


class BranchesAndBoundsMethod:
    def __init__(self, f, value):
        self.integer_solutions = []
        self.root = TreeNode(f, value)

    @staticmethod
    def find_float_idx(arr):
        if arr[0] == 'Нет решения':
            return False, 0, 0
        for idx, el in enumerate(arr[1:]):
            if not int(el) == float(el):
                return True, idx, math.floor(el)
        return False, 0, 0

    def branching(self, node):
        find, idx, el = self.find_float_idx(node.value.answer[:4])
        if find:
            # Ветвление влево если найдено дробное решение
            new_row = numpy.zeros(node.value.c.size)
            new_row[idx] = 1
            a = numpy.vstack((node.value.A, new_row))
            b = numpy.append(node.value.b, el)
            simplex = SimplexMethod(a, b, node.value.c, mode=MaxTowarding())
            try:
                print(f'ВЕТВЛЕНИЕ ВЛЕВО ПО ПЕРМЕННОЙ x_{idx} <= {el}')
                simplex.get_result()
            except AssertionError:
                node.left = TreeNode('Нет\n решения', simplex)
                print(f'в ветке x_{idx} <= {el} нет решения')
                return
            node.left = TreeNode(simplex.answer[0], simplex)
            self.branching(node.left)
            # Ветвление вправо если найдено дробное решение
            new_row_right = numpy.zeros(node.value.c.size)
            new_row_right[idx] = -1
            a_right = numpy.vstack((node.value.A, new_row_right))
            b_right = numpy.append(node.value.b, -(el + 1))
            simplex = SimplexMethod(a_right, b_right, node.value.c, mode=MaxTowarding())
            try:
                print(f'ВЕТВЛЕНИЕ ВПРАВО ПО ПЕРМЕННОЙ x_{idx} => {el+1}')
                simplex.get_result()
            except AssertionError:
                node.right = TreeNode('Нет решения', simplex)
                print(f'в ветке x_{idx} >= {el+1} нет решения')
                return
            node.right = TreeNode(simplex.answer[0], simplex)
            self.branching(node.right)
        if not find:
            if node.value.answer[0] == 'Нет\n решения':
                return
            print('Найдено целочисленное решение')
            self.integer_solutions.append(node)
            return

    def start(self):
        self.branching(self.root)
        print('Все целочисленные решения')
        for solution in self.integer_solutions:
            print(solution.value.answer[:4])
        self.print()

    def print(self):
        lines, *_ = self._make_string_representation(self.root)
        for line in lines:
            print(line)

    def _make_string_representation(self, node):
        """
        Создает рекурсивное представление дерева
        """
        # нет ни одного поддерева
        if node.right is None and node.left is None:
            line = '%s' % node.f
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # есть только левое поддерево
        if node.right is None:
            lines, n, p, x = self._make_string_representation(node.left)
            s = '%s' % node.f
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # есть только правое поддерево
        if node.left is None:
            lines, n, p, x = self._make_string_representation(node.right)
            s = '%s' % node.f
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # есть оба поддерева
        left, n, p, x = self._make_string_representation(node.left)
        right, m, q, y = self._make_string_representation(node.right)
        s = '%s' % node.f
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


def brute_force(a, b, c, optimal_value):
    """
    Функция полного перебора всех возможных целочисленных переменных
    :param a: Уравнения системы ограничений
    :param b: Массив ограничений
    :param c: Функционал
    :param optimal_value: Результат симплекс метода
    :return: Оптимальное целочисленное решение
    """
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)
    solutions = {}
    max_x = math.ceil(optimal_value / c[c > 0].min())

    for combination in itertools.product(numpy.arange(max_x), repeat=c.size):
        number_of_valid_constraints = 0
        for i in range(b.size):
            constraints = a[i] * combination
            if numpy.sum(constraints) <= b[i]:
                number_of_valid_constraints += 1

        if number_of_valid_constraints == b.size:
            result = numpy.sum(combination * c)
            solutions[result] = combination
            print(combination, result)

    return max(solutions.keys()), solutions[max(solutions.keys())]


if __name__ == '__main__':
    print('МЕТОД ВЕТВЕЙ И ГРАНИЦ')

    problem = Problem.from_yaml(sys.argv[1])

    simplex = SimplexMethod(problem.A, problem.b, problem.c, MaxTowarding())
    solution = simplex.get_result()

    tree = BranchesAndBoundsMethod(solution[0], simplex)
    tree.start()

    brute_solution, value = brute_force(problem.A, problem.b, problem.c, solution[0])
    print('ПОЛНЫЙ ПЕРЕБОР')
    print(f'F = {brute_solution}, x = {value}')
