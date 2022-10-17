import random
import math

class RandomPlayer:
    def __init__(self):
        self.number = None
    
    def set_player_number(self, n):
        self.number = n
    
    def choose_move(self, game_board, choices):
        return random.choice(choices)