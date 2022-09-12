import sys
sys.path.append('connect_four')
from game import *
from random_player import *
from input_player import *
from heuristic_minimax import *
import time

#game should be at max 10 sec
players = [HeuristicMiniMax(ply=2), RandomPlayer()]
game = ConnectFour(players)
game.run_to_completion()
print(f"Player {game.winner} won")