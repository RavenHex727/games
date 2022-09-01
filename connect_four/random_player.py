import random
import math

class RandomPlayer:
    def __init__(self):
        self.number = None
    
    def set_player_number(self, n):
        self.number = n
    
    def choose_move(self, game_board):
        choices = []

        for i in range(6):
            for j in range(7):
                if game_board[i][j] == 0 and j not in choices:
                    choices.append(j)

        return random.choice(choices)