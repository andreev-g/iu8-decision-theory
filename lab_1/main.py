import sys
import numpy as np

from src.simplex import (
    SimplexTable,
    SimplexProblem
)
from src.simplex.simplex_problem import FuncTarget


def main():
    p = SimplexProblem.from_yaml(sys.argv[1])
    p.to_canonical()
    matrix = p.get_matrix()
    simplex = SimplexTable(
        canonical_start_table=matrix,
        target=p.target
    )
    simplex.print()
    simplex.find_base_solution(inplace=True)
    simplex.print()
    print("find optimal")
    simplex.find_optimal_solution(inplace=True)
    simplex.print()


if __name__ == '__main__':
    main()
