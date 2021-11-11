import sys

from lab_1.src.problem import Problem
from lab_1.src.simplex import SimplexTable


def draw_delimiter(space_start=True, space_end=True):
    line_len = 100
    if space_start:
        print()
    # print("╦" * line_len)
    # print("╩" * line_len)
    print("▓" * line_len)
    if space_end:
        print()


if __name__ == '__main__':
    input_data_path = sys.argv[1]
    calc_dual = len(sys.argv) > 2

    problem = Problem.from_yaml(input_data_path)
    print("Задача:")
    problem.draw()
    draw_delimiter()

    problem = problem.to_canonical()
    print("Канонический вид задачи:")
    problem.draw()
    draw_delimiter()

    simplex = SimplexTable(problem)
    print("Начальная симплекс таблица выглядит следующим образом:")
    print()
    simplex.draw_table()
    print("Исходное значение переменных:")
    simplex.draw_solution()
    draw_delimiter()

    print("Найдем опорное решение:")
    print()
    if simplex.is_bearing_resolve():
        print("Данная таблица уже содержит опорное решение. Т.к. в столбце si0 все элементы >= 0")
    else:
        simplex.find_bearing_resolve()
        print("Найденное опорное решение")
        simplex.draw_table()
    draw_delimiter()

    print("Найдем оптимальное решение:")
    print()
    simplex.find_optimal_resolve()
    draw_delimiter()

    print("Оптимальное решение:")
    print()
    simplex.draw_solution()
    draw_delimiter()

    print("Проверка решения:")
    print()
    simplex.check_solution()
    draw_delimiter(space_end=False)

    if calc_dual:
        draw_delimiter(space_end=False, space_start=False)
        draw_delimiter(space_start=False)
        print("Проверим результат на двойственной задаче\n")

        problem = Problem.from_yaml(input_data_path).to_dual_problem()
        print("Двойственная задача:")
        problem.draw()
        draw_delimiter()

        problem = problem.to_canonical()
        print("Канонический вид задачи:")
        problem.draw()
        draw_delimiter()

        simplex = SimplexTable(problem)

        print("Найдем опорное решение:")
        print()
        if simplex.is_bearing_resolve():
            print("Данная таблица уже содержит опорное решение. Т.к. в столбце si0 все элементы >= 0")
        else:
            simplex.find_bearing_resolve()
            print("Найденное опорное решение")
            simplex.draw_table()
        draw_delimiter()

        print("Найдем оптимальное решение:")
        print()
        simplex.find_optimal_resolve()
        draw_delimiter()

        print("Оптимальное решение:")
        print()
        simplex.draw_solution()
        draw_delimiter()

        print("Проверка решения:")
        print()
        simplex.check_solution()
        draw_delimiter()
