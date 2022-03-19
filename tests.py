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


for n in range(1, 7, 2):
    print(n)
