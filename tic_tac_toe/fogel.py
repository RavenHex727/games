import math
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import sys
sys.path.append('tic_tac_toe')
from game import *
sys.path.append("tic_tac_toe/players")
from nn_player import *
from near_perfect import *
from input_player import *


logs = Logger('/workspace/games/fogel.txt')

def activation_function(x):
    return 1 / (1 + math.exp(-x))


class Node:
    def __init__(self, node_num):
        self.node_num = node_num
        self.parents = []
        self.node_input = None
        self.node_output = 0


class EvolvedNeuralNet:
    def __init__(self, nodes, node_weights, bias_node_nums, h):
        self.nodes = nodes
        self.node_weights = node_weights
        self.bias_nodes = bias_node_nums
        self.hidden_layer_size = h

        for bias_node_num in self.bias_nodes:
            self.nodes[bias_node_num - 1].node_output = 1

        for weight in node_weights:
            i, n = weight.split(',')
            current_node = self.get_node(int(i))
            next_node = self.get_node(int(n))
            next_node.parents.append(current_node)

    def build_neural_net(self, input_array):
        for n in range(9):
            self.nodes[n].node_input = input_array[n]
            self.nodes[n].node_output = input_array[n]

        for n in range(len(self.nodes)):
            node = self.nodes[n]

            if node.node_num in self.bias_nodes or n in [n for n in range(9)]:
                continue

            total_input = 0

            for input_node in node.parents:
                total_input += input_node.node_output * self.node_weights[str(input_node.node_num) + ',' + str(node.node_num)]

            node.node_input = total_input

            node.node_output = activation_function(total_input)

        return [node.node_output for node in self.nodes[-9:]]

    def get_node_outputs(self):
        info_dict = {}

        for node in self.nodes:
            info_dict[node.node_num] = node.node_output

        return info_dict

    def get_node(self, node_num):
        for node in self.nodes:
            if node.node_num == node_num:
                return node


def get_weight_ids(nodes, bias_node_nums, h):
    weight_ids = []
    nodes_by_layer = {1: [], 2: [], 3: []}

    for n in range(0, len(nodes)):
        node = nodes[n]

        if n <= 9:
            nodes_by_layer[1].append(node)

        elif node in nodes[-9:]:
            nodes_by_layer[3].append(node)

        else:
            nodes_by_layer[2].append(node)

    for layer in nodes_by_layer:
        if layer != list(nodes_by_layer.keys())[-1]:
            for node in nodes_by_layer[layer]:
                for next_layer_node in nodes_by_layer[layer + 1]:
                    if next_layer_node.node_num not in bias_node_nums:
                        weight_ids.append(f'{node.node_num},{next_layer_node.node_num}')

    return weight_ids


def make_new_gen_v2(parents):
    new_gen = copy.deepcopy(parents)

    for parent in parents:
        child_weights = {}
        child_hidden_layer_size = parent.hidden_layer_size
        child_bias_node_nums = parent.bias_nodes
        child_nodes = copy.deepcopy(parent.nodes)

        if random.randint(0, 1) == 0:
            add_subtract = random.choice(["Add", "Delete"])

            if add_subtract == "Add" and child_hidden_layer_size != 10:
                child_hidden_layer_size += 1
                child_nodes.insert(10, Node(len(parent.nodes) + 1))

            if add_subtract == "Subtract" and child_hidden_layer_size != 1:
                child_hidden_layer_size -= 1
                selected_node_num = random.choice([parent.nodes[n].node_num for n in range(10, 10 + parent.hidden_layer_size + 1)])

                for node in child_nodes:
                    if node.node_num == selected_node_num:
                        child_nodes.remove(node)

        child_weight_ids = get_weight_ids(child_nodes, child_bias_node_nums, child_hidden_layer_size)

        for weight in child_weight_ids:
            if weight in list(parent.node_weights.keys()):
                weight_value = parent.node_weights[weight] + np.random.normal(0, 0.05)
                assert abs(weight_value) - abs(parent.node_weights[weight]) < 0.3, "Child weight value changed too much"
                child_weights[weight] = parent.node_weights[weight] + np.random.normal(0, 0.05)

            else:
                child_weights[weight] = 0

        child = EvolvedNeuralNet(child_nodes, child_weights, child_bias_node_nums, child_hidden_layer_size)
        assert child != parent, "Child neural net is the same as parent"
        new_gen.append(child)

    return new_gen


def run_game(players):
    game = TicTacToe(players)
    game.run_to_completion()


def make_first_gen(population_size):
    first_gen = []

    for n in range(population_size):
        h = random.randint(1, 10)
        nodes = [Node(n) for n in range(1, 10 + h + 1 + 9)]
        weight_ids = get_weight_ids(nodes, [10, 10 + h + 1], h)
        weights = {}

        for weight_id in weight_ids:
            weight = random.uniform(-0.5, 0.5)
            assert abs(weight) <= 0.5
            weights[weight_id] = weight	

        neural_net = EvolvedNeuralNet(nodes, weights, [10, 10 + h + 1], h)
        first_gen.append(neural_net)

    return first_gen


def run_games(players, num_games):
    win_data = {1: 0, 2: 0, "Tie": 0}

    for _ in range(num_games):
        game = TicTacToe(players)
        game.run_to_completion()
        win_data[game.winner] += 1

    return win_data


def first_evaluation(neural_nets):
    payoff_data = {}

    for neural_net in neural_nets:
        payoff_data[neural_net] = 0

    for neural_net in neural_nets:
        win_data = run_games([NNPlayer(neural_net), NearPerfect()], 32)

        payoff_data[neural_net] += win_data[1] - 10 * win_data[2]

    return payoff_data


def get_subset(choices, excluded_nets, max_elements):
    subset = []
    choices.remove(excluded_nets[0])

    if len(choices) >= 10:
        while len(subset) < 10:
            random_net = random.choice(choices)

            if random_net not in subset and random_net not in excluded_nets:
                subset.append(random_net)
                excluded_nets.append(random_net)

    else:
        return choices

    return subset


def second_evaluation(payoff_data):
    for neural_net in list(payoff_data.keys()):
        comparing_nets = get_subset(list(payoff_data.keys()), [neural_net], 10)

        for net in comparing_nets:
            if payoff_data[neural_net] > payoff_data[net]:
                payoff_data[neural_net] += 1

            if payoff_data[neural_net] < payoff_data[net]:
                payoff_data[neural_net] -= 1

    return payoff_data 


def select_parents(payoff_data):
    sorted_data = sorted(payoff_data.items(), key=lambda x: x[1], reverse=True)
    sorted_nets = [info[0] for info in sorted_data]
    return sorted_nets[:int(len(sorted_nets) / 2)] #manually change this each time

'''
    max_payoff_value = -99999999

    for neural_net in payoff_data:
        if payoff_data[neural_net] > max_payoff_value:
            max_payoff_value = payoff_data[neural_net]
    
    next_gen_parents = []

    for neural_net in payoff_data:
        if payoff_data[neural_net] == max_payoff_value:
            next_gen_parents.append(neural_net)

    return next_gen_parents
'''

def find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, return_net=False):
    max_total_payoff_net = list(first_evaluation_data.keys())[0]

    for neural_net in first_evaluation_data:
        if first_evaluation_data[neural_net] > first_evaluation_data[max_total_payoff_net]:
            max_total_payoff_net = neural_net

    if return_net == False:
        return first_evaluation_data[max_total_payoff_net]
    
    else:
        return max_total_payoff_net


def select_parents_v2(payoff_data):
    max_payoff_value = -999999999999999999999999

    for neural_net in list(payoff_data.keys()):
        if payoff_data[neural_net] > max_payoff_value:
            max_payoff_value = payoff_data[neural_net]

    parents = []

    for neural_net in list(payoff_data.keys()):
        if payoff_data[neural_net] == max_payoff_value:
            parents.append(neural_net)

    return parents


file_object = open('neural_nets.txt', 'a')


def run(num_first_gen, num_gen):
    max_payoff_values = {}

    start_time = time.time()
    first_gen = make_first_gen(num_first_gen)
    first_evaluation_data = first_evaluation(first_gen)
    #print("First evaluation for Gen 0 Done")
    second_evaluation_data = second_evaluation(first_evaluation_data)
    #print("Second evaluation for Gen 0 Done")
    #testing next_gen_parents = select_parents(second_evaluation_data)
    next_gen_parents = select_parents(second_evaluation_data)
    #print("Parents from Gen 0 have been selected")
    max_payoff_values[0] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data)
    #print("Got Max Total Payoff Value for Gen 0")
    current_gen = make_new_gen_v2(next_gen_parents)
    print(f"Gen 0 took {time.time() - start_time} seconds to complete")


    for n in range(1, num_gen):
        start_time = time.time()
        first_evaluation_data = first_evaluation(current_gen)
        #print(f"First evaluation for Gen {n} Done")
        second_evaluation_data = second_evaluation(first_evaluation_data)
        #print(f"Second evaluation for Gen {n} Done")
        #testing next_gen_parents = select_parents(second_evaluation_data)
        next_gen_parents = select_parents(second_evaluation_data)
        #print(f"Parents from Gen {n} have been selected")

        if n == num_gen - 1:
            max_payoff_net = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, True)
            file_object.write(f'{max_payoff_net.__dict__} \n')
            max_payoff_values[n] = first_evaluation_data[max_payoff_net]

        max_payoff_values[n] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, False)

        #print(f"Got Max Total Payoff Value for Gen {n}")
        current_gen = make_new_gen_v2(next_gen_parents)
        print(f"Gen {n} took {time.time() - start_time} seconds to complete")

    return max_payoff_values

#code smth up to get neural net in last gen with highest 1st evaluation payoff
total_values = {}
first_gen_size = 50
num_generations = 100
num_trials = 20

#logs.write(f'HYPERPARAMETERS \n\t Networks in first generation: {first_gen_size} \n\t Selection percentage: 0.5')

for n in range(0, num_generations):
    total_values[n] = 0


for n in range(0, num_trials):
    start_time = time.time()
    max_payoff_values = run(first_gen_size, num_generations)

    for layer in max_payoff_values:
        total_values[layer] += max_payoff_values[layer]

    print(f"Trial {n} took {time.time() - start_time} seconds to complete")


x_values = [key for key in list(total_values.keys())]
y_values = [value / num_trials for value in list(total_values.values())]

plt.style.use('bmh')
plt.plot(x_values, y_values)
plt.xlabel('num generations')
plt.ylabel('max total payoff')
plt.legend(loc="best")
plt.savefig('fogel.png')