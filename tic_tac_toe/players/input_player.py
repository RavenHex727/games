import random
import math

class InputPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
    
    def set_player_symbol(self, n):
        self.symbol = n
    
    def set_player_number(self, n):
        self.number = n

    def get_possible_moves(self, board):
        possible_moves = [(i,j) for i in range(3) for j in range(3) if board[i][j] == 0]
        return possible_moves
    
    def choose_move(self, game_board):
        choices = [(i,j) for i in range(len(game_board)) for j in range(len(game_board)) if game_board[i][j]==0]
        print(game_board[0], "\n", game_board[1], "\n", game_board[2], "\n", choices , "\n")
        chosen_choice = input("Choose index of choice ")
        return choices[int(chosen_choice)]