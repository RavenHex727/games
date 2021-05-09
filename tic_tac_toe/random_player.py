from random import random
import math

class RandomPlayer():
    def __init__(self):
        self.player_num = None
    
    def set_player_numbers(self, n):
        self.player_num = n
    
    def choose_translation(self, options):
        random_idx = math.floor(len(options) * random())
        return options[random_idx]
