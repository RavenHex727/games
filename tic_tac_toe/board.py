def find_winner(input_list):
    winner = input_list[0]

    for n in input_list:
        if n != winner:
            return False

    return winner

def get_board_elements(board):
    board_elements = []

    for row in board:
        for value in row:
            board_elements.append(value)

    return board_elements

def check_for_winner(board):
        rows_columns_diagonals = get_rows(board) + get_columns(board) + get_diagonals(board)

        for element in rows_columns_diagonals:
            if find_winner(element) == 1:
                return 1

            if find_winner(element) == 2:
                return 2

        if None not in get_board_elements(board):
            return "Tie"

        return None


def get_rows(board):
    return [row for row in board]


def get_columns(board):
    columns = []

    for column_index in range(len(board[0])):
        columns.append([row[column_index] for row in board])

    return columns


def get_diagonals(board):
    diagonal1 = []
    upper_left_corner = (0, 0)
    diagonal2 = []
    upper_right_corner = (0, 2)

    for n in range(len(board[0])):
        diagonal1.append(board[upper_left_corner[0] + n][upper_left_corner[1] + n])
        diagonal2.append(board[upper_right_corner[0] + n][upper_right_corner[1] - n])

    return [diagonal1, diagonal2]


def make_board_from_state(state):
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for n in range(len(state)):
        if n < 3:
            board[0][n] = int(state[n])

        elif n < 6:
            board[1][n - 3] = int(state[n])

        elif n < 9:
            board[1][n - 6] = int(state[n])

    return board


def get_win_capture_frequency(strategy):
    can_win = 0
    will_win = 0

    for state in strategy:
        board = make_board_from_state(state)

        sum_rows = [sum(row) for row in get_rows(board) if 2 not in row]
        sum_columns = [sum(column) for column in get_columns(board) if 2 not in column]
        sum_diagonals = [sum(diagonals) for diagonals in get_diagonals(board) if 2 not in diagonals]

        if 2 in sum_rows or 2 in sum_columns or 2 in sum_diagonals:
            can_win += 1

        move = strategy[state]
        winner = check_for_winner(board)

        if move < 3:
            board[0][move] = strategy[state]

        elif move < 6:
            board[1][move - 3] = strategy[state]

        elif move < 9:
            board[1][move - 6] = strategy[state]

        if check_for_winner(board) == 1:
            will_win += 1

    return will_win / can_win


def get_loss_prevention_frequency(strategy):
    can_lose = 0
    can_block = 0

    for state in strategy:
        board = make_board_from_state(state)

        sum_rows = [sum(row) for row in get_rows(board) if 1 not in row]
        sum_columns = [sum(column) for column in get_columns(board) if 1 not in column]
        sum_diagonals = [sum(diagonals) for diagonals in get_diagonals(board) if 1 not in diagonals]

        if 4 in sum_rows or 4 in sum_columns or 4 in sum_diagonals:
            can_lose += 1

        move = strategy[state]
        winner = check_for_winner(board)

        if move < 3:
            board[0][move] = strategy[state]

        elif move < 6:
            board[1][move - 3] = strategy[state]

        elif move < 9:
            board[1][move - 6] = strategy[state]

        for row in get_rows(board):
            if sum([player_num for player_num in row if player_num != 1]) == 4 and 1 in row:
                can_block += 1

        for column in get_columns(board):
            if sum([player_num for player_num in column if player_num != 1]) == 4 and 1 in column:
                can_block += 1

        for diagonal in get_diagonals(board):
            if sum([player_num for player_num in diagonal if player_num != 1]) == 4 and 1 in diagonal:
                can_block += 1

    return can_block / can_lose