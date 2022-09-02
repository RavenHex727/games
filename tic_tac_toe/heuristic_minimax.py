import random
import math
import sys
sys.path.append('tic_tac_toe')
from reduced_depth_game_tree import *

class HeuristicMiniMax:
    def __init__(self, ply):
        self.symbol = None
        self.number = None
        self.ply = ply

    def set_player_symbol(self, n):
        self.symbol = n
  
    def set_player_number(self, n):
        self.number = n
        root_state = [[None, None, None], [None, None, None], [None, None, None]]
        self.game_tree = ReducedSearchGameTree(root_state, self.number, self.ply)

    def choose_move(self, game_board):
        choices = [(i,j) for i in range(3) for j in range(3) if game_board[i][j] == None]
        self.game_tree.reset_node_values()

        if game_board not in list(self.game_tree.nodes_dict.keys()):
            self.game_tree.nodes_dict[str(game_board)] = Node(game_board, self.number, self.number)

        current_node = self.game_tree.nodes_dict[str(game_board)]

        self.game_tree.build_tree([current_node])
        children = self.game_tree.build_tree([current_node])

        for _ in range(self.ply - 1):
            self.game_tree.build_tree(children)
            children = self.game_tree.build_tree(children)

        self.game_tree.set_node_values(current_node)
        max_value_node = current_node.children[0]

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