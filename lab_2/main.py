import sys

from src.simplex import (
    SimplexTable,
    SimplexProblem,
    demonstrate_base_solution,
    demonstrate_optimal_solution
)
from src.utility import (
    get_simplex,
    print_great_separator
)


def main():
    # simplex = get_simplex(sys.argv[1])
    # print_great_separator()
    # demonstrate_base_solution(simplex)
    # print_great_separator()
    # demonstrate_optimal_solution(simplex)
    # print_great_separator()

    simplex = get_simplex(sys.argv[2])
    print_great_separator()
    demonstrate_base_solution(simplex)
    print_great_separator()
    demonstrate_optimal_solution(simplex)
    print_great_separator()


if __name__ == '__main__':
    main()
