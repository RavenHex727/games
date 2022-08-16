class TicTacToe():
    def __init__(self, players):
        self.players = players
        self.board = [[None, None, None], [None, None, None], [None, None, None]]
        self.game_state = None
        self.set_player_numbers()
        self.turn = 0
        self.winner = None

    def get_game_state_arr(self):
        self.game_state = []

        for n in range(0, len(self.board)):
            for entry in self.board[n]:
                if entry == None:
                    self.game_state.append(0)

                else: 
                    self.game_state.append(entry)

        return self.game_state

    def set_player_numbers(self):
        for i, player in enumerate(self.players):
            player.set_player_numbers(i + 1)

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

        for n in range(len(self.board[0])):
            diagonal1.append(self.board[upper_left_corner[0] + n][upper_left_corner[1] + n])
            diagonal2.append(self.board[upper_right_corner[0] + n][upper_right_corner[1] - n])

        return [diagonal1, diagonal2]

    def get_board_elements(self):
        board_elements = []

        for row in self.board:
            for value in row:
                board_elements.append(value)

        return board_elements

    def find_winner(self, input_list):
        winner = input_list[0]

        for n in input_list:
            if n != winner:
                return False

        return winner

    def get_free_locations(self):
        available_locs = []

        for row_index in range(len(self.board)):
            for column_index in range(len(self.board[0])):
                if self.board[row_index][column_index] == None:
                    available_locs.append((row_index, column_index))

        return available_locs

    def check_for_winner(self):
        rows_columns_diagonals = self.get_rows() + self.get_columns() + self.get_diagonals()

        for element in rows_columns_diagonals:
            if self.find_winner(element) == 1:
                return 1

            if self.find_winner(element) == 2:
                return 2

        if None not in self.get_board_elements():
            return "Tie"

        return None

    def print_board(self):
        print("\n-----------")

        for row in self.board:
            for element in row[:-1]:
                print(element, end="  ")

            print(row[-1])

        print("-----------")

    def complete_turn(self):
        for player in self.players:
            possible_translations = self.get_free_locations()
            choice = player.choose_move(self.get_game_state_arr())
            self.board[choice[0]][choice[1]] = player.player_num

            if self.check_for_winner() != None:
                self.winner = self.check_for_winner()
                break

        self.turn += 1

    def run_to_completion(self):
        while self.winner == None:
            for player in self.players:
                player.get_game_state(self.get_game_state_arr())

            self.complete_turn()
