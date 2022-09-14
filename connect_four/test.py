import sys
sys.path.append('connect_four')
from game import *
from random_player import *
from input_player import *
from heuristic_minimax import *
import time

#game should be at max 10 sec
'''
win_data = {1: 0, 2: 0, "Tie": 0}

for _ in range(20):
    players = [HeuristicMiniMax(ply=4), RandomPlayer()]
    game = ConnectFour(players)
    game.run_to_completion()
    win_data[game.winner] += 1

for _ in range(20):
    players = [RandomPlayer(), HeuristicMiniMax(ply=4)]
    game = ConnectFour(players)
    game.run_to_completion()

    if game.winner != "Tie":
        win_data[3 - game.winner] += 1

    else:
        win_data["Tie"] += 1

print(win_data)
'''

'''
avg_time = 0
for _ in range(25):
    start_time = time.time()
    players = [RandomPlayer(), HeuristicMiniMax(ply=4)]
    game = ConnectFour(players)
    game.run_to_completion()
    avg_time += time.time() - start_time
    print(time.time() - start_time)

print(f"avg time is {avg_time/25}")
'''
players = [InputPlayer(), HeuristicMiniMax(ply=4)]
game = ConnectFour(players)
game.run_to_completion()
print(game.winner)