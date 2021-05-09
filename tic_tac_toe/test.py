import sys
sys.path.append('tic_tac_toe')
from game import *
from random_player import *

alternate = False
win_data = {1: 0, 2: 0}

for _ in range(100):
    game = TicTacToe([RandomPlayer(), RandomPlayer()])
    game.run_to_completion()

    if game.winner != 'Tie':
        if alternate:
            win_data[3 - game.winner] += 1

        else:
            win_data[game.winner] += 1

    alternate = not alternate

print("100 games:", win_data, "\n")

alternate = False
win_data = {1: 0, 2: 0}

for _ in range(1000):
    players = [RandomPlayer(), RandomPlayer()]
    game = TicTacToe(players)
    game.run_to_completion()

    if game.winner != 'Tie':
        if alternate:
            win_data[3 - game.winner] += 1

        else:
            win_data[game.winner] += 1

    alternate = not alternate

print("1000 games:", win_data)

