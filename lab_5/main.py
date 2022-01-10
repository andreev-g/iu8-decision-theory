from lab_5.game import PlayersGame

if __name__ == '__main__':
    # вариант 25
    matrix = [
        [15, 12, 7, 2, 3],
        [4, 10, 18, 4, 15],
        [16, 13, 19, 3, 19],
        [12, 1, 1, 19, 12],
    ]

    game = PlayersGame(matrix)
    game.solve()
    game.print_solution()
