class Board():
    def __init__(self, state, board=None):
        self.state = state
        self.board = board

    def find_winner(self, input_list):
        winner = input_list[0]

        for n in input_list:
            if n != winner:
                return False

        return winner

    def get_board_elements(self):
        board_elements = []

        for row in self.board:
            for value in row:
                board_elements.append(value)

        return board_elements

    def check_for_winner(self):
            rows_columns_diagonals = get_rows(self.board) + get_columns(self.board) + get_diagonals(self.board)

            for element in rows_columns_diagonals:
                if self.find_winner(element) == 1:
                    return 1

                if self.find_winner(element) == 2:
                    return 2

            if None not in get_board_elements():
                return "Tie"

            return None


    def get_rows(self):
        return [row for row in self.board]


    def get_columns(self):
        columns = []

        for column_index in range(len(self.board[0])):
            columns.append([row[column_index] for row in self.board])

        return columns


    def get_diagonals(self):
        diagonal1 = []
        upper_left_corner = (0, 0)
        diagonal2 = []
        upper_right_corner = (0, 2)

        for n in range(len(board[0])):
            diagonal1.append(self.board[upper_left_corner[0] + n][upper_left_corner[1] + n])
            diagonal2.append(self.board[upper_right_corner[0] + n][upper_right_corner[1] - n])

        return [diagonal1, diagonal2]


    def make_board_from_state(self, state):
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