import sys
import yaml
import enum
import math
import numpy
import pydantic
import itertools
import typing as t

from src.simplex.v2 import SimplexMethod


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


class SysTowarding(str, enum.Enum):
    MIN = "min"
    MAX = "max"


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
            simplex = SimplexMethod(a, b, node.value.c, mode=SysTowarding.MAX)
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
            simplex = SimplexMethod(a_right, b_right, node.value.c, mode=SysTowarding.MAX)
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
        lines, *_ = self._new_string_representation(self.root)
        for line in lines:
            print(line)

    def _new_string_representation(self, node):
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
            lines, n, p, x = self._new_string_representation(node.left)
            s = '%s' % node.f
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # есть только правое поддерево
        if node.left is None:
            lines, n, p, x = self._new_string_representation(node.right)
            s = '%s' % node.f
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # есть оба поддерева
        left, n, p, x = self._new_string_representation(node.left)
        right, m, q, y = self._new_string_representation(node.right)
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

    simplex = SimplexMethod(problem.A, problem.b, problem.c, SysTowarding.MAX)
    solution = simplex.get_result()

    tree = BranchesAndBoundsMethod(solution[0], simplex)
    tree.start()

    # brute_solution, value = brute_force(problem.A, problem.b, problem.c, solution[0])
    # print('ПОЛНЫЙ ПЕРЕБОР')
    # print(f'F = {brute_solution}, x = {value}')
