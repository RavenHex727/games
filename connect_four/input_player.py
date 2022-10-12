import random
import math

class InputPlayer:
    def __init__(self):
        self.number = None
    
    def set_player_number(self, n):
        self.number = n
    
    def choose_move(self, game_board, choices):
        for row in game_board:
            print(row)

        choice = input(f"Which column do you wish to make your move in? These are your options: {choices} \n")

        return int(choice)