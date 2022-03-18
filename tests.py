import random
import time

def get_game_state(board):
    game_state = ''
    board_info = []

    for n in range(0, len(board)):
        for entry in board[n]:
            game_state += str(entry)

    return game_state
start_time = time.time()

board = [[random.randint(0, 1), random.randint(0, 1) ,random.randint(0, 1) ], [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)], [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]]
test_board = [[1, 0, 2], [2, 1, 0], [1, 1, 1]]

test_arr = [n for n in range(25)]

print(test_arr[:5])