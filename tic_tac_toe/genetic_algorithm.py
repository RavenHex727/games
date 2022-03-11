import sys
sys.path.append('tic_tac_toe')
from game import *
from random_player import *
import time
import random
from itertools import combinations

'''
file_object = open('all_possible_game_states.txt', 'a')

all_possible_game_states = []

def get_random_game_state():
    state = ''

    for _ in range(9):
        state += str(random.choice([0, 1, 2]))

    return state

start_time = time.time()

while len(all_possible_game_states) < (3 ** 9 - 2 ** 9):
    new_state = get_random_game_state()

    if new_state not in all_possible_game_states and '0' in new_state:
        all_possible_game_states.append(new_state)


for state in all_possible_game_states:
    file_object.write(state + '\n')

'''

file = open("all_possible_game_states.txt", 'r')
content_of_file = file.read()

all_game_states = content_of_file.split("\n")


def get_random_board_index(game_state):
    open_spaces = []

    for n in range(0, len(game_state)):
        if game_state[n] == '0':
            open_spaces.append(n)

    return random.choice(open_spaces)

'''
all_strategies = {}

for n in range(25):
    strategy = {}

    for state in all_game_states:
        strategy[state] = get_random_board_index(state)

    all_strategies[n + 1] = RandomPlayer(strategy)

all_players = []

for index in all_strategies:
    all_players.append(all_strategies[index])
'''


def get_all_match_ups(players):
    all_matchups = []
    
    for i in combinations(players, 2):
        all_matchups.append(i)

    return all_matchups


def run_match(players):
    win_data = {players[0]: 0, players[1]: 0, "Tie": 0}

    game = TicTacToe(players)
    game.run_to_completion()

    if game.winner == 1:
        win_data[players[0]] += 1
    
    elif game.winner == 2:
        win_data[players[1]] += 1
    
    elif game.winner == "Tie":
        win_data["Tie"] += 1

    game = TicTacToe(players[::-1])
    game.run_to_completion()

    if game.winner == "Tie":
        win_data["Tie"] += 1
    
    elif game.winner == 1:
        win_data[players[1]] += 1

    elif game.winner == 2:
        win_data[players[0]] += 1

    return win_data


def get_optimal_strategies(strategies):
    strategies_arr = []
    optimal_strategies = []

    for strategy in strategies:
        strategies_arr.append((strategy, strategies[strategy]))

    sorted_strategies = sorted(strategies_arr, key=lambda x: x[1])[::-1][:5]

    for strategy in sorted_strategies:
        optimal_strategies.append(strategy[0])

    return optimal_strategies


def mate(strategies):
    children = [strategy for strategy in strategies]
    pairings = get_all_match_ups(strategies)
    base_strat = {}

    for state in all_game_states:
        base_strat[state] = None

    for parents in pairings:
        child1 = {}
        child2 = {}

        for state in all_game_states:

            child1[state] = random.choice([parents[0].strategy[state], parents[1].strategy[state]])
            child2[state] = random.choice([parents[0].strategy[state], parents[1].strategy[state]])

        children.append(RandomPlayer(child1))
        children.append(RandomPlayer(child2))


    return children


first_generation = []

for n in range(25):
    strategy = {}

    for state in all_game_states:
        strategy[state] = get_random_board_index(state)

    first_generation.append(RandomPlayer(strategy))


total_scores = {}
all_matchups = get_all_match_ups(first_generation)

for player in first_generation:
    total_scores[player] = 0

for matchup in all_matchups:
    win_data = run_match(matchup)

    for player in matchup:
        for outcome in win_data:
            if outcome == "Tie":
                total_scores[player] += 0

            elif outcome != player:
                total_scores[player] -= win_data[outcome]

            elif outcome == player:
                total_scores[player] += win_data[outcome]

optimal_strategies = get_optimal_strategies(total_scores)

next_generation = mate(optimal_strategies)