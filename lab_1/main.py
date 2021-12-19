import sys

from lab_1.src import (
    demonstrate_base_solution,
    demonstrate_optimal_solution
)
from lab_1.src.utility import print_great_separator
from lab_1.src.simplex.simplex_table import SimplexTable
from lab_1.src.simplex.simplex_problem import SimplexProblem


def main():

    problem = SimplexProblem.from_yaml(sys.argv[1])
    simplex = SimplexTable(problem)
    print("Исходная таблица\n")
    simplex.print()
    print_great_separator()

    demonstrate_base_solution(simplex)
    print_great_separator()

    demonstrate_optimal_solution(simplex)
    print_great_separator()


if __name__ == '__main__':
    main()
