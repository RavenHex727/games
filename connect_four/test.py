import sys
sys.path.append('connect_four')
from game import *
from random_player import *
from input_player import *
from semi_random_player import *
from heuristic_minimax import *
sys.path.append('connect_four/competition_stuff')
#from cayden_comp import *
from maia import *
from justin import *
from charlie import *
from anton import *
from william import *
from cayden_comp import *
import time


def run_games(n, players):
    win_data = {1: 0, 2: 0, "Tie": 0}

    for _ in range(int(n / 2)):
        game = ConnectFour(players)
        game.run_to_completion()
        win_data[game.winner] += 1

    for _ in range(int(n / 2)):
        game = ConnectFour(players[::-1])
        game.run_to_completion()

        if game.winner != "Tie":
            win_data[3 - game.winner] += 1

        else:
            win_data["Tie"] += 1

    return win_data


print(f"WW VS MD {run_games(4, [William(), Maia()])}")

'''
players = [Justin(), SemiRandomPlayer()]
game = ConnectFour(players)
game.run_to_completion()
print(game.winner)
'''