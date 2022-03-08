import sys
sys.path.append('tic_tac_toe')
from game import *
from random_player import *

all_possible_game_states = []

while all_possible_game_states < (3 ** 9):
    for state in all_possible_game_states:
        if len(state) != 