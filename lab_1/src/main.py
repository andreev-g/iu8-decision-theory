from os import path

from lab_1.src.problem import Problem
from lab_1.src.simplex import SimplexTable


if __name__ == '__main__':
    input_data_path = f"{path.dirname(__file__)}/../input_data.yaml"
    problem = Problem.from_yaml(input_data_path)
    problem = problem.to_canonical()

    table = SimplexTable(problem)
    table.is_bearing_resolve()
    print(table._table)
    table.find_bearing_resolve()
    table.find_optimal_resolve()
    print(table._table)
    print(table.get_solution())
