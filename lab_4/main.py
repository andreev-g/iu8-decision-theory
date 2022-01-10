from src.brute import BruteForce
from src.utility import print_separator

from lab_4.gomory import Gomory


if __name__ == '__main__':
    # вариант 25
    A = [[2, 1, 1],
         [1, 4, 0],
         [0, 0.5, 1]]
    c = [8, 6, 2]
    b = [4, 3, 6]

    gomory_method = Gomory(A, b, c)
    solution = gomory_method.solution()

    print_separator()
    print('Полный перебор:')
    print_separator()
    bf = BruteForce(A, b, c, solution[0])
    brute_solution, value = bf.brute_optimal()
    print("Оптимальное решение полным перебором:")
    print(f'F = {brute_solution}, x = {value}')
