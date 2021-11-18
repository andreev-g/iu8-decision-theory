import sys

import numpy as np

from src.simplex import (
    SimplexTable,
    SimplexProblem
)


def main():
    p = SimplexProblem.from_yaml(sys.argv[1])
    p.to_canonical()
    matrix = p.get_matrix()
    matrix = np.array([
        [2,  1,  -2],
        [-2, -2,  1],
        [5,  1,   1],
        [0,  1,  -1],
    ])
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
