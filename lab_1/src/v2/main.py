from lab_1.src.v2.simplex import SimplexTable


def main():
    simplex = SimplexTable(
        canonical_start_table=[
            [4, 2, 1,   1],
            [3, 1, 4,   0],
            [6, 0, 0.5, 1],
            [0, 8, 6,   2],
        ],
        # canonical_start_table=[
        #     [2,  1,  -2],
        #     [-2, -2, 1],
        #     [5,  1,  1],
        #     [0,  1,  -1],
        # ],
        target="min"
    )
    simplex.print()
    simplex.find_base_solution(inplace=True)
    simplex.print()
    simplex.find_optimal_solution(inplace=True)
    simplex.print()


if __name__ == '__main__':
    main()
