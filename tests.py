import random

def get_game_state(board):
    game_state = ''
    board_info = []

    for n in range(0, len(board)):
        for entry in board[n]:
            game_state += str(entry)

    return game_state


board = [[random.randint(0, 1), random.randint(0, 1) ,random.randint(0, 1) ], [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)], [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]]
test_board = [[1, 0, 2], [2, 1, 0], [1, 1, 1]]

print(get_game_state(test_board))