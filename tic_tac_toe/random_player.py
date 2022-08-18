import random
import math

class RandomPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
    
    def set_player_symbol(self, n):
        self.symbol = n
    
    def set_player_number(self, n):
        self.number = n
    
    def get_possible_moves(self, board):
        possible_moves = [(i,j) for i in range(3) for j in range(3) if board[i][j] == None]
        return possible_moves
    
    def choose_move(self, game_board):
        choices = [(i,j) for i in range(len(game_board)) for j in range(len(game_board)) if game_board[i][j]==None]
        random_idx = math.floor(len(choices) * random.random())
        return choices[random_idx]

class TestPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
    
    def set_player_symbol(self, n):
        self.symbol = n
    
    def set_player_number(self, n):
        self.number = n
    
    def get_possible_moves(self, board):
        possible_moves = [(i,j) for i in range(3) for j in range(3) if board[i][j] == None]
        return possible_moves
    
    def choose_move(self, game_board):
        choices = self.get_possible_moves(game_board)

        for i in range(0, len(game_board)):
            if None in game_board[i]:
                top_most_row = i

        for j in range(0, len(game_board[top_most_row])):
            if game_board[top_most_row][j] == None:
                left_most = j


        return (top_most_row, left_most)