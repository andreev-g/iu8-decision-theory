import math
import numpy as np

from src.brute import BruteForce
from src.utility import print_separator
from src.simplex.simplex import SimplexMethod


class TreeNode:
    def __init__(self, f, value: SimplexMethod):
        self.left = None
        self.right = None
        self.f = f
        self.value = value


class MinTowarding:
    def __str__(self):
        return 'min'


class MaxTowarding:
    def __str__(self):
        return 'max'


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
            new_row = np.zeros(node.value.c.size)
            new_row[idx] = 1
            a = np.vstack((node.value.A, new_row))
            b = np.append(node.value.b, el)
            simplex = SimplexMethod(a, b, node.value.c, mode=MaxTowarding())
            try:
                print(f'Ветвимся влево по переменной x_{idx} <= {el}')
                simplex.get_result()
            except AssertionError:
                node.left = TreeNode('Нет\n решения', simplex)
                print(f'В ветви x_{idx} <= {el} нет решения')
                return
            node.left = TreeNode(simplex.answer[0], simplex)
            self.branching(node.left)
            # Ветвление вправо если найдено дробное решение
            new_row_right = np.zeros(node.value.c.size)
            new_row_right[idx] = -1
            a_right = np.vstack((node.value.A, new_row_right))
            b_right = np.append(node.value.b, -(el + 1))
            simplex = SimplexMethod(a_right, b_right, node.value.c, mode=MaxTowarding())
            try:
                print(f'Ветвимся вправо по переменной x_{idx} => {el+1}')
                simplex.get_result()
            except AssertionError:
                node.right = TreeNode('Нет решения', simplex)
                print(f'В ветви x_{idx} >= {el+1} нет решения')
                return
            node.right = TreeNode(simplex.answer[0], simplex)
            self.branching(node.right)
        if not find:
            if node.value.answer[0] == 'Нет\n решения':
                return
            print('Найдено целочисленное решение:')
            self.integer_solutions.append(node)
            return

    def start(self):
        self.branching(self.root)
        print_separator()
        print('Все целочисленные решения')
        for solution in self.integer_solutions:
            print(f"F({solution.value.answer[1]}, {solution.value.answer[2]}, {solution.value.answer[3]}) = {solution.value.answer[0]}")
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


if __name__ == '__main__':
    print('Метод ветвей и границ:')
    print_separator()

    # c = [2, 8, 3]
    # A = [[2, 1, 1],
    #      [1, 2, 0],
    #      [0, 0.5, 1]]
    # b = [4, 6, 2]

    c = [3, 3, 7]
    A = [[1, 1, 1],
         [1, 4, 0],
         [0, 0.5, 3]]
    b = [3, 5, 7]

    simplex = SimplexMethod(A, b, c, MaxTowarding())
    solution = simplex.get_result()

    tree = BranchesAndBoundsMethod(solution[0], simplex)
    tree.start()

    print_separator()
    print('Полный перебор:')
    print_separator()
    bf = BruteForce(A, b, c, solution[0])
    brute_solution, value = bf.brute_optimal()
    print("Оптимальное решение полным перебором:")
    print(f'F = {brute_solution}, x = {value}')
