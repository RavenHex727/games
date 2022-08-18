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
        #print("Choices", choices)

        max_value_node = current_node.children[0]
        debug_info = {}

        for child in current_node.children:
            debug_info[str(child.state)] = child.value

        #print(debug_info)

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
        #print("Choice:", choice)
        return choice

'''
    def get_move_from_boards(self, base_state, new_state):
        base_state_children = self.game_tree.nodes_dict[str(base_state)].children

        for i in range(len(new_state)):
            for j in range(len(new_state[0])):
                base = base_state[i][j]
                new = new_state[i][j]
                if base != new:
                    return (i, j)

    def choose_move(self, game_board):
        base_state = self.game_tree.root_node.state
        node_values = [node.value for node in self.game_tree.root_node.children]
        max_value_index = node_values.index(max(node_values))
        best_move_node = self.root_node.children[max_index]
        new_state = best_move_node.state
        return self.get_move_from_boards(base_state, new_state)
'''  
        