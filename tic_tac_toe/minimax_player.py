from random import random
import math

class MiniMaxPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
  
    def set_player_symbol(self, n):
        self.symbol = n
  
    def set_player_number(self, n):
        self.number = n
    
    def choose_move(self, choices):
        random_idx = math.floor(len(choices) * random())
        return choices[random_idx]