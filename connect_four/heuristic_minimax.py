import random
import math
import sys
sys.path.append('connect_four')
#from reduced_depth_game_tree import *
from game_tree import *


class HeuristicMiniMax:
    def __init__(self, ply):
        self.symbol = None
        self.number = None
        self.ply = ply

    def set_player_symbol(self, n):
        self.symbol = n
  
    def set_player_number(self, n):
        self.number = n
        root_state = [[0 for _ in range(7)] for _ in range(6)]
        self.game_tree = ReducedSearchGameTree(root_state, self.number, self.ply)

    def get_row_with_lowest_available_column(self, j, board):
        largest_row = 0

        for n in range(len(board)):
            if board[n][j] == 0:
                largest_row = n

        return largest_row

    def evaluate_optimal_choices(self, game_board, optimal_choices):
        info = {}

        for choice in optimal_choices:
            new_board = copy.deepcopy(game_board)
            i = self.get_row_with_lowest_available_column(choice, new_board)
            new_board[i][choice] = self.number

            current_node = self.game_tree.nodes_dict[str(new_board)]
            info[choice] = current_node.heuristic_evaluation()
            #work in progress

    def choose_move(self, game_board):
        choices = []

        for i in range(6):
            for j in range(7):
                if game_board[i][j] == 0 and j not in choices:
                    choices.append(j)

        self.game_tree.reset_node_values()

        if game_board not in list(self.game_tree.nodes_dict.keys()):
            self.game_tree.nodes_dict[str(game_board)] = Node(game_board, self.number, self.number)

        current_node = self.game_tree.nodes_dict[str(game_board)]
        self.game_tree.build_ply(current_node)
 
        self.game_tree.set_node_values(current_node)
        max_value_node = current_node.children[0]

        for child in current_node.children:
            if child.value > max_value_node.value:
                max_value_node = child

        optimal_choices = []

        for choice in choices:
            new_board = copy.deepcopy(game_board)
            i = self.get_row_with_lowest_available_column(choice, new_board)
            new_board[i][choice] = self.number

            if self.game_tree.nodes_dict[str(new_board)].check_for_winner() == self.number:
                return choice

            if self.game_tree.nodes_dict[str(new_board)].value == max_value_node.value:
                optimal_choices.append(choice)

        choice = random.choice(optimal_choices)
        return choice