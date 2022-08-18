from random import random
from logger import *

class TicTacToe:
    def __init__(self, players):
        self.players = players
        self.set_player_symbols()
        self.set_player_numbers()
        self.logs = Logger('/workspace/games/tic_tac_toe/logs.txt')
        self.logs.clear_log()
        #self.determine_player_order()
        self.board = [[None for _ in range(3)] for _ in range(3)]
        self.round =  1
        self.winner = None
        self.log_board()
  
    def set_player_symbols(self): 
        self.players[0].set_player_symbol('X')
        self.players[1].set_player_symbol('O')

    def set_player_numbers(self): 
        self.players[0].set_player_number(1)
        self.players[1].set_player_number(2)
  
    #def determine_player_order(self):
        #rand = round(random())

        #if rand == 1:
            #self.players = self.players[::-1]

    def player_num_board(self):
        board_copy = [[None, None, None], [None, None, None], [None, None, None]]

        for i in range(0, len(self.board)):
            for j in range(0, len(self.board[i])):
                if self.board[i][j] == "X":
                    board_copy[i][j] = 1

                if self.board[i][j] == "O":
                    board_copy[i][j] = 2

        return board_copy

    def complete_round(self):
        for player in self.players:
            player_move = player.choose_move(self.player_num_board())
            self.board[player_move[0]][player_move[1]] = player.symbol

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

    def check_for_winner(self):
        rows = self.board.copy()
        cols = [[self.board[i][j] for i in range(3)] for j in range(3)]
        diags = [[self.board[i][i] for i in range(3)],
             [self.board[i][2-i] for i in range(3)]]

        board_full = True
        for row in rows + cols + diags:
            if None in row:
                board_full = False

            for player in self.players:
                if row == [player.symbol for _ in range(3)]:
                    return player.number
    
        if board_full:
            return 'Tie'

        return None

    def log_board(self):
        for i in range(len(self.board)):
            row = self.board[i]
            row_string = ''

            for space in row:
                if space == None:
                    row_string += '_|'

                else:
                    row_string += space + '|'

            self.logs.write(row_string[:-1] + "\n")

        self.logs.write('\n')