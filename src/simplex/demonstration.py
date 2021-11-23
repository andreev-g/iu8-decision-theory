from src.simplex.simplex_table import SimplexTable


def demonstrate_base_solution(simplex: SimplexTable) -> None:
    print((" " * 9) + "Нахождение опорного решения:\n")
    if simplex.is_base_solution():
        print("Данная таблица уже содержит опорное решение")
        return
    simplex.find_base_solution(inplace=True, print_logs=True)


def demonstrate_optimal_solution(simplex: SimplexTable) -> None:
    print("Нахождение оптимального решения")
    simplex.find_optimal_solution(inplace=True, print_logs=True)
    solution = simplex.get_solution()
    dict_solution = {
        f"x{i + 1}": round(value, 3)
        for i, value in enumerate(solution)
    }
    print(f"Найденное решение: {dict_solution}")
    print()
    if not simplex.check_solution():
        raise ValueError("\nРешение неверно!")
    print("\nНайденное решение корректно")
