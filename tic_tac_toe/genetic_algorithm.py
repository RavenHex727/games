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


def run_tourney_match(players):
    game = TicTacToe(players)
    game.run_to_completion()
    winner = None

    if game.winner == 1:
        winner = players[0]
    
    elif game.winner == 2:
        winner = players[1]
    
    elif game.winner == "Tie":
        winner = None

    return winner

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

def run_matches(match_ups, players):
    scores = {}

    for player in players:
        scores[player] = 0

    for match in match_ups:
        win_data = run_match(match)

        for player in match:
            for outcome in win_data:
                if outcome == "Tie":
                    scores[player] += 0

                elif outcome != player:
                    scores[player] -= win_data[outcome]

                elif outcome == player:
                    scores[player] += win_data[outcome]

    return scores


def get_sub_set(exclude_elements, space, size):
    subset = []

    while len(subset) < size:
        element = random.choice(space)

        if element not in exclude_elements and element not in subset:
            subset.append(element)

    return subset

def stochastic_rr(strategies, n):
    all_matchups = get_all_match_ups(strategies)
    total_scores = run_matches(all_matchups, strategies)
    selected_strats = []

    while len(selected_strats) < n/4:
        strategies_arr = []
        subset = get_sub_set(selected_strats, strategies, n/8)

        for strategy in subset:
            strategies_arr.append((strategy, total_scores[strategy]))

        sorted_strategies = sorted(strategies_arr, key=lambda x: x[1])[::-1]

        selected_strat = sorted_strategies[0][0]
        selected_strats.append(selected_strat)

    return selected_strats


def tournament_selection_rr(strategies, n):
    best_players = []
    all_strats = strategies.copy()

    while len(best_players) < n / 4:
        subset = get_sub_set(best_players, all_strats, n/8)
        matches = get_all_match_ups(subset)
        scores = run_matches(matches, subset)

        best_player = get_top_n_strategies(scores, 1)
        best_players.append(best_player)

    return best_players


def get_top_n_strategies(scores, n):
    strategies_arr = []
    optimal_strategies = []

    for strategy in scores:
        strategies_arr.append((strategy, scores[strategy]))

    sorted_strategies = sorted(strategies_arr, key=lambda x: x[1])[::-1][:int(n)]

    if n == 1:
        return sorted_strategies[0][0]

    for strategy in sorted_strategies:
        optimal_strategies.append(strategy[0])

    return optimal_strategies


def gene_selection(mutation_rate, parents, state):
    if random.randint(1, 100) < mutation_rate * 100:
        return get_random_board_index(state)

    else:
        return random.choice([parents[0].strategy[state], parents[1].strategy[state]])


def mate(strategies, mutation_rate, population_size):
    children = [strategy for strategy in strategies]

    while len(children) < population_size:
        parents = []

        while len(parents) < 2:
            parent = random.choice(strategies)
            
            if parent not in parents:
                parents.append(parent)

        child1 = {}

        for state in all_game_states:
            child1[state] = gene_selection(mutation_rate, parents, state)

        children.append(RandomPlayer(child1))

    return children


first_generation = []

for n in range(32):
    strategy = {}

    for state in all_game_states:
        strategy[state] = get_random_board_index(state)

    first_generation.append(RandomPlayer(strategy))


def run_bracket(strategies):
    scores = {}
    random.shuffle(strategies)

    for strat in strategies:
        scores[strat] = 1

    current_round = strategies

    while len(current_round) > 0:
        next_round = []

        for n in range(0, int(len(current_round)) - 1, 2):
            matchup = [current_round[n], current_round[n + 1]]

            winner = run_tourney_match(matchup)

            if winner == None:
                winner = random.choice(matchup)

            scores[winner] += 1
            next_round.append(winner)

        random.shuffle(next_round)

        current_round = next_round
        next_round = []

    return scores


def hardcutoff_bracket(current_gen, n):
    scores = run_bracket(current_gen)
    optimal_strategies = get_top_n_strategies(scores, n/4)
    return optimal_strategies


def stochastic_bracket(strategies, n):
    scores = run_bracket(strategies)
    selected_strats = []

    while len(selected_strats) < n/4:
        strategies_arr = []
        subset = get_sub_set(selected_strats, strategies, n/8)

        for strategy in subset:
            strategies_arr.append((strategy, scores[strategy]))

        sorted_strategies = sorted(strategies_arr, key=lambda x: x[1])[::-1]

        selected_strat = sorted_strategies[0][0]
        selected_strats.append(selected_strat)

    return selected_strats


def tournament_bracket(strategies, n):
    best_players = []
    all_strats = strategies.copy()

    while len(best_players) < n / 4:
        subset = get_sub_set(best_players, all_strats, n/8)
        scores = run_bracket(subset)

        best_player = get_top_n_strategies(scores, 1)
        best_players.append(best_player)

    return best_players


def hard_cutoff_rr(current_gen, n):
    all_matchups = get_all_match_ups(current_gen)
    total_scores = run_matches(all_matchups, current_gen)
    optimal_strategies = get_top_n_strategies(total_scores, n/4)

    return optimal_strategies


def compare_to_gen(comparison_gen, top_strats):
    total_scores = {}

    for player in top_strats:
        total_scores[player] = 0

    for strat in top_strats:
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


def run(selection_method, fitness_score, mutation_rate, n):
    avg_score_vs_gen1 = []
    avg_score_vs_previous_gen = []
    selected_strats = None

    current_gen = first_generation

    previous_generation = first_generation

    if selection_method == "hard cutoff" and fitness_score == "round robin":
        selected_strats = hard_cutoff_rr(current_gen, n)

    if selection_method == "tournament" and fitness_score == "round robin":
        selected_strats = tournament_selection_rr(current_gen, n)

    if selection_method == "stochastic" and fitness_score == "round robin":
        selected_strats = stochastic_rr(current_gen, n)

    if selection_method == "tournament" and fitness_score == "bracket":
        selected_strats = tournament_bracket(current_gen, n)

    if selection_method == "stochastic" and fitness_score == "bracket":
        selected_strats = stochastic_bracket(current_gen, n)

    if selection_method == "hard cutoff" and fitness_score == "bracket":
        selected_strats = hardcutoff_bracket(current_gen, n)

    win_caps = []
    lost_prev = []
    #win_cap_single = [get_win_capture_frequency(strat.strategy) for strat in selected_strats]
    lost_prev_single = [get_loss_prevention_frequency(strat.strategy) for strat in selected_strats]
    #win_caps.append(sum(win_cap_single)/ len(win_cap_single))
    lost_prev.append(sum(lost_prev_single) / len(lost_prev_single))
    #avg_score_vs_gen1.append(compare_to_gen(first_generation, first_generation))
    #avg_score_vs_previous_gen.append(compare_to_gen(first_generation, first_generation))

    for _ in range(24):
        current_gen = mate(selected_strats, mutation_rate, n)

        if selection_method == "hard cutoff" and fitness_score == "round robin":
            selected_strats = hard_cutoff_rr(current_gen, n)

        if selection_method == "tournament" and fitness_score == "round robin":
            selected_strats = tournament_selection_rr(current_gen, n)

        if selection_method == "stochastic" and fitness_score == "round robin":
            selected_strats = stochastic_rr(current_gen, n)

        if selection_method == "tournament" and fitness_score == "bracket":
            selected_strats = tournament_bracket(current_gen, n)

        if selection_method == "stochastic" and fitness_score == "bracket":
            selected_strats = stochastic_bracket(current_gen, n)

        if selection_method == "hard cutoff" and fitness_score == "bracket":
            selected_strats = hardcutoff_bracket(current_gen, n)

        #win_cap_single = [get_win_capture_frequency(strat.strategy) for strat in selected_strats]
        lost_prev_single = [get_loss_prevention_frequency(strat.strategy) for strat in selected_strats]
        #win_caps.append(sum(win_cap_single)/ len(win_cap_single))
        lost_prev.append(sum(lost_prev_single) / len(lost_prev_single))
        #avg_score_vs_gen1.append(compare_to_gen(first_generation, selected_strats))
        #avg_score_vs_previous_gen.append(compare_to_gen(previous_generation, selected_strats))
        previous_generation = current_gen

    return lost_prev
    #{"Vs 1st Gen": avg_score_vs_gen1, "Vs Prev Gen": avg_score_vs_previous_gen, "Win Caps": win_caps, "Loss Prevs": lost_prev}


start_time = time.time()
plt.style.use('bmh')
plt.plot([n for n in range(25)], [n for n in run("hard cutoff", "round robin", 0.001, 32)], label="hard cutoff round robin")
plt.plot([n for n in range(25)], [n for n in run("tournament", "round robin", 0.001, 32)], label="tournament round robin")
plt.plot([n for n in range(25)], [n for n in run("stochastic", "round robin", 0.001, 32)], label="stochastic round robin")
plt.plot([n for n in range(25)], [n for n in run("hard cutoff", "bracket", 0.001, 32)], label="hard cutoff bracket")
plt.plot([n for n in range(25)], [n for n in run("tournament", "bracket", 0.001, 32)], label="tournament bracket")
plt.plot([n for n in range(25)], [n for n in run("stochastic", "bracket", 0.001, 32)], label="stochastic bracket")
plt.xlabel('# generations completed')
plt.ylabel('avg lost_prev frequencies')
plt.legend(loc="best")
plt.savefig('lost_prev.png')

print(time.time() - start_time)




'''
plt.clf()
plt.plot([n for n in range(35)], [n for n in avg_score_vs_previous_gen])
plt.xlabel('# generations completed')
plt.ylabel('avg total score of tournament selected 5 strats')
plt.savefig('vs_prev_gen.png')

plt.clf()
plt.plot([n for n in range(35)], [n for n in win_caps])
plt.xlabel('# generations completed')
plt.ylabel('avg loss prev freq')
plt.savefig('loss_prevent.png')

plt.clf()
plt.plot([n for n in range(35)], [n for n in lost_prev])
plt.xlabel('# generations completed')
plt.ylabel('avg win cap freq')
plt.savefig('win_caps.png')
'''