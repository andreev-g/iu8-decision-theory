import sys

from src.simplex import (
    SimplexTable,
    demonstrate_base_solution,
    demonstrate_optimal_solution
)
from src.utility import (
    get_simplex,
    print_great_separator
)


def make_dual_simplex(simplex: SimplexTable) -> SimplexTable:
    return simplex


def main():
    simplex = get_simplex(sys.argv[1])
    print_great_separator()
    demonstrate_base_solution(simplex)
    print_great_separator()
    demonstrate_optimal_solution(simplex)
    print_great_separator()


if __name__ == '__main__':
    main()
