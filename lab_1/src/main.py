from os import path

from lab_1.src.problem import Problem
from lab_1.src.simplex import SimplexTable


def draw_delimiter():
    line_len = 100
    print()
    print("╦" * line_len)
    print("╩" * line_len)
    print()


if __name__ == '__main__':
    input_data_path = f"{path.dirname(__file__)}/../input_data.yaml"

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
    draw_delimiter()
