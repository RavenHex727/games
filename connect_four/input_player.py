import random
import math

class InputPlayer:
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

        for row in game_board:
            print(row)

        choice = input(f"Which column do you wish to make your move in? These are your options: {choices} \n")

        if int(choice) not in choices:
            print("Invalid move, try again")
            choice = input(f"Which column do you wish to make your move in? These are your options: {choices}")

        return int(choice)