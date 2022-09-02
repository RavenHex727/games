import random
import time
import numpy as np

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

def get_sub_set(exclude_elements, space, size):
    subset = []

    while len(subset) < size:
        element = random.choice(space)

        if element not in exclude_elements and element not in subset:
            subset.append(element)

    return subset


sets = []
used_nums = []
while len(sets) < 5:
    subset = get_sub_set(used_nums, test_arr, 2)

    sets.append(subset)

    for n in subset:
        used_nums.append(n)


board = [[random.choice([0, 1, 2]) for _ in range(7)] for _ in range(6)]

def log_board(board):
    for i in range(len(board)):
        row = board[i]
        row_string = ''

        for space in row:
            if space == "0":
                row_string += ' '

            else:
                row_string += str(space) + ' '

        print(row_string[:-1])

    print('\n')

board = [[1, 2, 3, 4, 5, 6, 7], 
        [8, 9, 10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19, 20, 21], 
        [22, 23, 24, 25, 26, 27, 28], 
        [29, 30, 31, 32, 33, 34, 35], 
        [36, 37, 38, 39, 40, 41, 42]]

max_col = len(board[0])
max_row = len(board)
cols = [[] for _ in range(max_col)]
rows = [[] for _ in range(max_row)]
fdiag = [[] for _ in range(max_row + max_col - 1)]
bdiag = [[] for _ in range(len(fdiag))]
min_bdiag = -max_row + 1

for x in range(max_col):
    for y in range(max_row):
        cols[x].append(board[y][x])
        rows[y].append(board[y][x])
        fdiag[x+y].append(board[y][x])
        bdiag[x-y-min_bdiag].append(board[y][x])


print([n for n in range(3)])