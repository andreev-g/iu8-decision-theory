import numpy
import itertools

from src.brute import BruteForce
from src.utility import print_separator
from lab_4.simplex import SimplexMethod


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

    A = [[1, 1, 1],
         [1, 4, 0],
         [0, 0.5, 3]]
    c = [3, 3, 7]
    b = [3, 5, 7]

    gomory_method = Gomory(A, b, c, 'max')
    solution = gomory_method.solve()

    print_separator()
    print('Полный перебор:')
    print_separator()
    bf = BruteForce(A, b, c, solution[0])
    brute_solution, value = bf.brute_optimal()
    print("Оптимальное решение полным перебором:")
    print(f'F = {brute_solution}, x = {value}')
