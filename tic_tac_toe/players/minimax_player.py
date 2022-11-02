import random
import math
import sys
sys.path.append('tic_tac_toe')
from game_tree import *

class MiniMaxPlayer:
    def __init__(self):
        self.symbol = None
        self.number = None
  
    def set_player_symbol(self, n):
        self.symbol = n
  
    def set_player_number(self, n):
        self.number = n
        root_state = [[None, None, None], [None, None, None], [None, None, None]]
        self.game_tree = GameTree(root_state, self.number)
        self.game_tree.build_tree()
        self.game_tree.set_node_values()

    def choose_move(self, game_board):
        choices = [(i,j) for i in range(3) for j in range(3) if game_board[i][j] == None]
        current_node = self.game_tree.nodes_dict[str(game_board)]
        max_value_node = current_node.children[0]
        debug_info = {}

        for child in current_node.children:
            debug_info[str(child.state)] = child.value

        for child in current_node.children:
            if child.value > max_value_node.value:
                max_value_node = child

        optimal_choices = []

        for choice in choices:
            new_board = copy.deepcopy(game_board)
            new_board[choice[0]][choice[1]] = self.number

            if new_board == max_value_node.state:
                optimal_choices.append(choice)
        
        choice = random.choice(optimal_choices)
        return choice