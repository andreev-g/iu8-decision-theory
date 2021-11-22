from src.simplex.simplex_table import SimplexTable
from src.simplex.simplex_problem import SimplexProblem


def print_great_separator() -> None:
    print("╦" * 70)
    print("╩" * 70)


def get_simplex(config_filename) -> SimplexTable:
    p = SimplexProblem.from_yaml(config_filename)
    matrix = p.get_matrix()
    simplex = SimplexTable(
        canonical_start_table=matrix,
        target=p.target,
        c=p.c
    )
    print((" " * 9) + "Исходная таблица")
    simplex.print()
    return simplex
