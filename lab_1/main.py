import sys

from src.simplex import (
    SimplexTable,
    SimplexProblem
)


def print_great_separator() -> None:
    print("╦" * 70)
    print("╩" * 70)


def get_simplex() -> SimplexTable:
    p = SimplexProblem.from_yaml(sys.argv[1])
    p.to_canonical()
    matrix = p.get_matrix()
    simplex = SimplexTable(
        canonical_start_table=matrix,
        target=p.target
    )
    print((" " * 9) + "Исходная таблица")
    simplex.print()
    return simplex


def demonstrate_base_solution(simplex: SimplexTable) -> None:
    print((" " * 9) + "Нахождение опорного решения:\n")
    if simplex.is_base_solution():
        print("Данная таблица уже содержит опорное решение")
        return
    simplex.find_base_solution(inplace=True, print_logs=True)


def demonstrate_optimal_solution(simplex: SimplexTable) -> None:
    simplex.find_optimal_solution(inplace=True, print_logs=True)
    simplex.check_solution()
    solution = simplex.get_solution()
    dict_solution = {
        f"x{i + 1}": round(value, 3)
        for i, value in enumerate(solution)
    }
    print(f"Найденное решение: {dict_solution}")


def main():
    simplex = get_simplex()
    print_great_separator()
    demonstrate_base_solution(simplex)
    print_great_separator()
    demonstrate_optimal_solution(simplex)
    print_great_separator()


if __name__ == '__main__':
    main()
