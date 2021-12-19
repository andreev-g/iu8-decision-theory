import itertools
from typing import List, Tuple
import numpy
from numpy.lib import math
import pandas


def print_separator():
    print('-' * 50)


class SimplexMethod:
    def __init__(self, a, b, c, mode):
        self.A = numpy.array(a)
        self.b = numpy.array(b)
        self.c = numpy.array(c)
        self.mode = mode

        # распологаем вектор свободных коэффициентов слева от матрицы A
        matr = numpy.c_[b, self.A]
        # распологаем вектор коэффициентов целевой функции в нижней строке симплекс-таблицы
        self.table = numpy.r_[matr, [[0, *self.c]]]
        # сохраняем исходную симплекс-таблицу для того, чтобы иметь возможность
        # проверить валидность решения в дальнейшем
        self.src_table = numpy.copy(self.table)
        # инвертируем знак коэффициентов целевой функции
        self.table[-1] *= -1

        # формируем легенды столбцев: [S0, x1, x2, x3]
        s0 = 'S0'
        self.columns = [s0] + [f'x_{i + 1}' for i in range(self.table.shape[1] - 1)]
        # формируем легенды строк: [F, x4, x5, x6]
        f = 'F'
        self.rows = [f'x_{i + 4}' for i in range(self.b.size)] + [f]

    def _exchange_basic_variable(self, resolving_element) -> numpy.ndarray:
        """
        Данный метод предназначен введения в базис переменной из столбца k вместо
        переменной из строки r, где r = resoling_element[0], k = resoling_element[1].
        Возвращаемым значением является симплекс-таблица с замененной базисной переменной
        """
        # переименовываем легенду базисной переменной, которую хотим заменить
        self.rows[resolving_element[0]] = self.columns[resolving_element[1]]
        # создаем матрицу, в которую будем записывать новую симплекс-таблицу
        new_matrix = numpy.zeros(self.table.shape)

        r = resolving_element[0]
        k = resolving_element[1]
        # проходим по всем элементам старой таблицы и формируем новую
        for i, j in numpy.ndindex(self.table.shape):
            if i == r:
                # попали на разрешающую строку
                # формула: s*[r][j] = s[r][j]/s[r][k]
                new_matrix[r, j] = self.table[r, j] / self.table[r, k]
            else:
                # попали на любой другой элемент
                # формула: s*[i][j] = s*[i][j] - (s[i][k] * s[r][j])/s[r][k]
                prod = (self.table[i, k] * self.table[r, j] / self.table[r, k])
                new_matrix[i, j] = self.table[i, j] - prod

        return new_matrix

    def _to_pivot_solution(self) -> List[Tuple[Tuple[int, int], pandas.DataFrame]]:
        """
        Данный метод преобразует симплекс-таблицу до опороного решения (по ходу
        преобразования понимаем совместна ли система).
        Метод возвращает ход решения опорного решения в виде массива вида:
        [( (r_0, k_0), simplex-table_0), ..., ( (r_n, k_n)), simplex-table_n)]
        """

        def find_resolving_element() -> Tuple[int, int]:
            """
            Функция ищет разрешающий элемент.
            Функция возвращает кортеж индексов разрешающего элемента в случае нахождения,
            иначе возвращает None
            """
            # получаем массив свободных коэффициентов (коэффициенты при ЦФ не рассматриваем)
            free_coefs = self.table[:-1, 0]
            # ищем индекс строки, в которой свобоный коэффициент < 0
            negative_row = numpy.where(free_coefs < 0, free_coefs, numpy.inf).argmin()
            # если такой строки нет, то разрешающего элемента нет
            if negative_row == 0 and free_coefs[negative_row] > 0:
                return None
            # получаем коэффициенты при переменных из строки, где есть своб. коэф. < 0
            row = self.table[negative_row, 1:]

            # проверяем на наличие решений (если нет коэффициента < 0, то решений нет)
            assert numpy.any(row < 0), 'система несовместна'
            # находим разрешающий столбец (там должен находиться коэффициет < 0)
            k = numpy.where(row < 0, row, numpy.inf).argmin()
            # увеличиваем k на 1, чтобы учесть наличие свободного коэффициента в таблице
            k += 1
            # получаем коэффициенты при переменных для разрешающего столбца (без ЦФ)
            resolving_column = self.table[:-1, k]
            # игнорируем возможные деления на 0 в контекстном менеджере
            with numpy.errstate(divide='ignore'):
                # делим соответствующие свобдные коэффициенты на коэффициенты
                # разрешающего столбца и записываем бесконечность в ячейки, где частное <= 0
                quotient = free_coefs / resolving_column
                quotient[quotient <= 0] = numpy.inf
                # берем индекс наименьшего положительного частного и проверяем, что он
                # действительно найден. В случае ненахождения считаем, что решений бесконечно
                r = quotient.argmin()
                assert quotient[r] != numpy.inf, f'cистема имеет бесконечно число решений'

            return (r, k)

        # массив с шагами нахождения опорного решения
        solution_progress = list()
        # преобразуем симплекс-таблицу до тех пор, пока не будет найдено опорное решение
        found_pivot_solution = False
        while not found_pivot_solution:
            # находим разрешающий элемент. Если его нет, то данное решение является опорным
            rk = find_resolving_element()
            if rk is None:
                found_pivot_solution = True
                continue

            # производим замен базисной переменной и добавляем запись в протокол решения
            self.table = self._exchange_basic_variable(rk)
            solution_progress.append((
                rk,
                pandas.DataFrame(
                    data=numpy.copy(self.table),
                    index=numpy.copy(self.rows),
                    columns=numpy.copy(self.columns)
                )
            ))

        return solution_progress

    def _find_optimal_solution(self) -> List[Tuple[Tuple[int, int], pandas.DataFrame]]:
        """
        Данный метод преобразует симплекс-таблицу до оптимального решения.
        Метод возвращает ход решения опорного решения в виде массива вида:
        [( (r_0, k_0), simplex-table_0), ..., ( (r_n, k_n)), simplex-table_n)]
        """

        def find_resolving_element() -> Tuple[int, int]:
            """
            Функция ищет разрешающий элемент.
            Функция возвращает кортеж индексов разрешающего элемента
            """
            k = 0
            if self.mode == 'min':
                # если мы минимизируем ЦФ, то ищем в коэффициентах ЦФ максимальный
                # в качестве разрешающего
                k = numpy.argmax(self.table[-1, :][1:]) + 1
            elif self.mode == 'max':
                # в случае максимизации - минимальный
                k = numpy.argmin(self.table[-1, :][1:]) + 1
            else:
                raise ValueError("mode could be 'max' or 'min' only")

            # получаем разрешающий столбец и стобец свободных коэффициентов
            resolving_column = self.table[:, k][:-1]
            free_coefs = self.table[:, 0][:-1]
            # игнорируем возможные деления на 0 в контекстном менеджере
            with numpy.errstate(divide='ignore'):
                # делим соответствующие свобдные коэффициенты на коэффициенты
                # разрешающего столбца и записываем бесконечность в ячейки, где частное <= 0
                quotient = free_coefs / resolving_column
                quotient[quotient < 0] = numpy.inf
                # берем индекс наименьшего положительного частного и проверяем, что он
                # действительно найден. В случае ненахождения считаем, что решений бесконечно
                r = quotient.argmin()
                assert quotient[r] != numpy.inf, f'Система имеет бесконечно много решений'
            return r, k

        # задаем условие остановки алгоритма: если минимизируем, то остановимся,
        # когда все коэффициенты при ЦФ <= 0, если максимизируем - все коэффициенты при ЦФ >= 0
        if self.mode == 'min':
            stop_condition = lambda: not all(i <= 0 for i in self.table[-1, 1:])
        elif self.mode == 'max':
            stop_condition = lambda: not all(i >= 0 for i in self.table[-1, 1:])

        # массив с шагами нахождения оптимального решения
        solution_progress = list()
        while stop_condition():
            # находим разрешающий элемент
            rk = find_resolving_element()

            # производим замен базисной переменной и добавляем запись в протокол решения
            self.table = self._exchange_basic_variable(rk)
            solution_progress.append((
                rk,
                pandas.DataFrame(
                    data=numpy.copy(self.table),
                    index=numpy.copy(self.rows),
                    columns=numpy.copy(self.columns)
                )
            ))

        return solution_progress

    def _verify_solution(self):
        """
        Данный метод проверяет правильно ли была вычислена ЦФ, а также смотрит не
        были ли были нарушены ограничения
        """
        # получаем решение задачи и коэффициенты при целевой функции
        solution = self._get_solution()
        f_coefs = self.src_table[-1, 1:4]
        # считаем значение целевой функции и сравниваем с найденным значением
        f = sum([f_coefs[idx] * var for idx, var in enumerate(solution[1:4])])
        assert solution[0] == f, f'Результат оптимального решения не совпадает с коэффициентами F={solution[0]}, f={f}'

        # итерируемся по ограничениям и проверяем соответствует ли им решение
        for number, limitation_conditions in enumerate(self.src_table[:-1]):
            prod = limitation_conditions[1:] * solution[1:]
            limit = numpy.sum(prod)
            assert limitation_conditions[
                       0] >= limit, f'Ограничение №{number + 1} нарушено: {limit} <= {limitation_conditions[0]}'
            print(f'Ограничение №{number + 1}: {limit} <= {limitation_conditions[0]}')

        print('Решение верно!')

    def _get_solution(self) -> numpy.ndarray:
        """
        Данный метод возвращает решение задачи в виде:
        [ F(X), x1, ..., xn ]
        """
        # создаем массив коэффициентов и кладем на первое место значение ЦФ
        solution = [self.table[-1, 0]]
        # добавляем коэффициенты базисных переменных, остальные коэффициенты полагаем равными нулю
        for var_number in range(1, self.table.shape[1]):
            if f'x_{var_number}' in self.rows:
                var = self.table[self.rows.index(f'x_{var_number}'), 0]
            else:
                var = 0
            solution.append(var)
        return numpy.array(solution)

    def solve(self):
        def print_progress(progress):
            """
            Функция печати ряда шагов (хода решений) симплекс-метода
            """
            for step in progress:
                print_separator()
                print('Индекс разрешающего элемента: ', step[0])
                print(step[1])

        print('СИМПЛЕКС МЕТОД')
        print_separator()
        print('Исходная симплекс-таблица:')
        print(pandas.DataFrame(data=self.table, index=self.rows, columns=self.columns))

        progress = self._to_pivot_solution()
        print_progress(progress)

        print('Поиск оптимального решения:')
        progress = self._find_optimal_solution()
        print_progress(progress)

        self._verify_solution()
        solution = self._get_solution()

        print('x1 =', solution[1])
        print('x2 =', solution[2])
        print('x3 =', solution[3])
        print('F =', solution[0])

        return self.table[:, 0]


class Gomory:
    def __init__(self, a, b, c, mode):
        self.a = numpy.array(a)
        self.b = numpy.array(b)
        self.c = numpy.array(c)
        self.mode = mode

        # дополняем матрицу ограничений A и вектор ЦФ фиктивными переменными
        # A, E -> A|E
        self.a = numpy.column_stack((self.a, numpy.eye(self.b.size)))
        # for example: c, (0, 0, 0) -> (c0, ..., ci, 0, 0, 0)
        self.c = numpy.append(self.c, numpy.zeros(self.b.size))

        # создаем объект класса SimplexMethod в котором и будут производится
        # все вычисления симплекс-метода
        self.simplex = SimplexMethod(self.a, self.b, self.c, self.mode)

    @staticmethod
    def _is_integer_solution(solution):
        """
        Данная функция проверяет является ли это решение целочисленным
        """
        for el in solution:
            if not el.is_integer():
                return False
        return True

    @staticmethod
    def _find_with_max_fractional_part(solution):
        """
        Данная функция получает значение максимальной дробной части
        """
        return (solution % 1).argmax()

    def solve(self):
        print('МЕТОД ОТСЕКАЮЩИХ ПЛОСКОСТЕЙ (МЕТОД ГОМОРИ)')

        found_interger_solution = False
        while not found_interger_solution:
            # находим оптимальное решение симплекс-методом и проверяем на целочисленность
            solution = self.simplex.solve()
            if self._is_integer_solution(solution[:-1]):
                found_interger_solution = True
                continue

            # ищем базисную переменную с наибольшей дробной частью
            idx = self._find_with_max_fractional_part(solution)
            # получаем массив ограничений, который состоит из дробныых частей всех
            # переменных в разложении переменной найденной на предыдущем шаге
            var_fractions = self.simplex.table[idx, 1:]
            var_fractions %= 1
            # дополняем снизу матрицу ограничений новым ограничением из дробных
            # частей, взятых сотрицательным знаком
            var_fractions *= -1
            a = numpy.vstack((self.simplex.A, var_fractions))
            # получаем значение свободного коэффициента при найденной переменной и
            # дополняем им вектор свобоных членов
            free_coef = -(solution[idx] % 1)
            b = numpy.append(self.simplex.b, free_coef)
            # создаем столбец для новой фиктивной переменной, соответсвующей найденному ограничению
            # и дополняем им справа симплекс-таблицу
            dummy_col = numpy.zeros(b.size)
            dummy_col[-1] = 1
            a = numpy.column_stack((a, dummy_col))
            # дополняем вектор коэффициентов ЦФ до количества переменных
            с = numpy.append(self.simplex.c, 0)
            # запускам сиплекс-метод с округлением всех чисел до 10 знаков во избежание зацикливания
            self.simplex = SimplexMethod(numpy.around(a, 10), numpy.around(b, 10), numpy.around(с, 10), self.mode)

        return solution


def brute_force(a, b, c, optimum):
    """
    Функция полного перебора всех возможных целочисленных переменных
    """
    print_separator()
    print('МЕТОД ПОЛНОГО ПЕРЕБОРА')
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array(c)
    solutions = {}

    var_limit = optimum / numpy.min(c)
    for combination in itertools.product(numpy.arange(var_limit), repeat=c.size):
        number_of_valid_constraints = 0
        for i in range(b.size):
            constraints = a[i] * combination
            if numpy.sum(constraints) <= b[i]:
                number_of_valid_constraints += 1

        if number_of_valid_constraints == b.size:
            result = numpy.sum(combination * c)
            solutions[result] = combination
            print(combination, result)

    optimal_solution = max(solutions.keys())
    return optimal_solution, solutions[optimal_solution]


if __name__ == '__main__':
    A = [[-2, -6],
         [-8, -3]]
    c = [1, 1]
    b = [-1, -1]

    s = SimplexMethod(A, b, c, 'max')
    s.solve()

    A = [[3, 1, 1],
         [1, 2, 0],
         [0, 0.5, 2]]
    c = [2, 6, 7]
    b = [3, 8, 1]

    gomory_method = Gomory(A, b, c, 'max')
    solution = gomory_method.solve()

    solution = brute_force(A, b, c, solution[-1])
    print('Решение:')
    print('x1 =', solution[1][0])
    print('x2 =', solution[1][1])
    print('x3 =', solution[1][2])
    print('F =', solution[0])
