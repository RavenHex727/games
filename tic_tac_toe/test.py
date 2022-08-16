import sys
sys.path.append('tic_tac_toe')
from game import *
from random_player import *

'''
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

'''

'''
all_possible_game_states = []

def get_random_game_state():
    state = ''

    for _ in range(9):
        state += str(random.choice([0, 1, 2]))

    return state

start_time = time.time()

while len(all_possible_game_states) < (3 ** 9):
    new_state = get_random_game_state()

    if new_state not in all_possible_game_states:
        all_possible_game_states.append(new_state)


for _ in range(10):
    game = TicTacToe([RandomPlayer(), RandomPlayer()])
    game.run_to_completion()
    print(game.winner)

'''

players = [RandomPlayer(), RandomPlayer()]
game = TicTacToe(players)
game.run_to_completion()
print(game.winner)