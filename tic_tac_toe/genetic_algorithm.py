import sys
sys.path.append('tic_tac_toe')
from game import *
from board import *
from random_player import *
import time, random, math
from itertools import combinations
import matplotlib.pyplot as plt

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


def get_current_gen_top_5(current_gen):
    total_scores = {}
    all_matchups = get_all_match_ups(current_gen)

    for player in current_gen:
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

    #next_generation = mate(optimal_strategies)
    return optimal_strategies


def compare_to_gen(comparison_gen, top5_strats):
    total_scores = {}

    for player in top5_strats:
        total_scores[player] = 0

    for strat in top5_strats:
        for old_strat in comparison_gen:
            win_data = run_match([strat, old_strat])

            for outcome in win_data:
                if outcome == "Tie":
                    total_scores[strat] += 0

                elif outcome != strat:
                    total_scores[strat] -= win_data[outcome]

                elif outcome == strat:
                    total_scores[strat] += win_data[outcome]

    scores = list(total_scores.values())

    return sum(scores) / len(scores)

avg_score_vs_gen1 = []
avg_score_vs_previous_gen = []


current_gen = first_generation
optimal_strats = get_current_gen_top_5(first_generation)
avg_score_vs_gen1.append(compare_to_gen(first_generation, first_generation))

for _ in range(30):
    current_gen = mate(optimal_strats)
    optimal_strats = get_current_gen_top_5(current_gen)
    avg_score_vs_gen1.append(compare_to_gen(first_generation, optimal_strats))

print(avg_score_vs_gen1)

'''

new_gen = first_generation
optimal_strats = get_current_gen_top_5(first_generation)
previous_generation = first_generation

for _ in range(20):
    new_gen = mate(optimal_strats)
    optimal_strats = get_current_gen_top_5(new_gen)
    avg_score_vs_previous_gen.append(compare_to_gen(previous_generation, optimal_strats))
    previous_generation = new_gen

print(avg_score_vs_previous_gen)
#plt.style.use('bmh')

'''
'''
plt.plot([n for n in range(31)], [n for n in avg_score_vs_gen1])
plt.xlabel('# generations completed')
plt.ylabel('average total score when playing against first generation')
plt.savefig('vs_1st_gen.png')
'''

'''
new_gen = first_generation
optimal_strats = get_current_gen_top_5(first_generation)
previous_generation = first_generation
win_cap_freq = []

for _ in range(10):
    new_gen = mate(optimal_strats)
    board = Board(test_strat)
    optimal_strats = get_current_gen_top_5(new_gen)
    win_cap_freq.append(get_win_capture_frequency(test_strat))
'''