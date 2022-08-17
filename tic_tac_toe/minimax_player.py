from random import random
import math
import sys
sys.path.append('tic_tac_toe')
from game_tree import *

class MiniMaxPlayer:
    def __init__(self, game_tree):
        self.symbol = None
        self.number = None
        self.game_tree = game_tree
  
    def set_player_symbol(self, n):
        self.symbol = n
  
    def set_player_number(self, n):
        self.number = n

    def set_node_values(self):
        current_nodes = []

        for node in game_tree.terminal_nodes:
            if node.winner == self.number:
                node.value = 1

            if node.winner != self.number and node.winner != "Tie":
                node.value = -1

            if node.winner == "Tie":
                node.value = 0

            current_nodes += node.previous

        while None not in current_nodes:
            for node in current_nodes:
                if node.turn == self.number:
                    node.value = max([child.value for child in node.children])

                if node.turn == 3 - self.number:
                    node.value = min([child.value for child in node.children])

                current_nodes.append(node.previous)
                current_nodes.remove(node)


    def choose_move(self, choices):
        self.set_node_values()

        for choice in choices:
            if choice[choice[0]][choice[1]] = 

        return choices[self.best_move_index()]