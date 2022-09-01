from random import random
from logger import *

class ConnectFour:
    def __init__(self, players):
        self.players = players
        self.set_player_numbers()
        self.logs = Logger('/workspace/games/connect_four/logs.txt')
        self.logs.clear_log()
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.round =  1
        self.winner = None
        self.log_board()

    def set_player_numbers(self): 
        self.players[0].set_player_number(1)
        self.players[1].set_player_number(2)

    def get_row_with_lowest_available_column(self, j):
        largest_row = 0

        for n in range(len(self.board)):
            if self.board[n][j] == 0:
                largest_row = n

        return largest_row

    def complete_round(self):
        for player in self.players:
            player_move = player.choose_move(self.board)
            i = self.get_row_with_lowest_available_column(player_move)
            self.board[i][player_move] = player.number

            if self.check_for_winner() != None:
                self.winner = self.check_for_winner()
                break

        self.round += 1
        self.log_board()

    def run_to_completion(self):
        while self.winner == None:
            self.complete_round()

        if self.winner != 'Tie':
            self.logs.write(f'Player {self.winner} won')

        else:
            self.logs.write('Tie')

    def get_diagonals(self):
        fdiag = [[] for _ in range(len(self.board) + len(self.board[0]) - 1)]
        bdiag = [[] for _ in range(len(fdiag))]

        for x in range(len(self.board[0])):
            for y in range(len(self.board)):
                fdiag[x + y].append(self.board[y][x])
                bdiag[x - y - (1 - len(self.board))].append(self.board[y][x])

        return fdiag + bdiag

    def check_for_winner(self):
        rows = self.board.copy()
        cols = [[self.board[i][j] for i in range(6)] for j in range(7)]
        diags = self.get_diagonals()

        str_info = []

        board_full = True

        for info in rows + cols + diags:
            if 0 in info:
                board_full = False

        for info in rows + cols + diags:
            for player in self.players:
                if str(player.number) * 4 in "".join([str(element) for element in info]):
                    return player.number

        if board_full:
            return 'Tie'

        return None

    def log_board(self):
        for i in range(len(self.board)):
            row = self.board[i]
            row_string = ''

            for space in row:
                if space == "0":
                    row_string += '0'

                else:
                    row_string += str(space) + ' '

            self.logs.write(row_string[:-1] + "\n")

        self.logs.write('\n')