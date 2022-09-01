import sys
sys.path.append('connect_four')
from game import *
from random_player import *
from input_player import *
import time

players = [InputPlayer(), RandomPlayer()]
game = ConnectFour(players)
game.run_to_completion()
print(f"Player {game.winner} won")