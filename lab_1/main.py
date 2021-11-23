import sys

from src.simplex import (
    demonstrate_base_solution,
    demonstrate_optimal_solution
)
from src.utility import print_great_separator
from src.simplex.simplex_table import SimplexTable
from src.simplex.simplex_problem import SimplexProblem


def main():

    problem = SimplexProblem.from_yaml(sys.argv[1])
    simplex = SimplexTable(problem)
    simplex.print()

    demonstrate_base_solution(simplex)
    print_great_separator()

    demonstrate_optimal_solution(simplex)
    print_great_separator()


if __name__ == '__main__':
    main()
