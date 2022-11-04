import math
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('tic_tac_toe')
from game import *
sys.path.append("tic_tac_toe/players")
from nn_player import *
from near_perfect import *
from input_player import *


def activation_function(x):
    return 1 / (1 + math.exp(-x))


class Node:
    def __init__(self, node_num):
        self.node_num = node_num
        self.parents = []
        self.node_input = None
        self.node_output = 0


class EvolvedNeuralNet:
    def __init__(self, num_nodes, node_weights, bias_node_nums, h):
        self.nodes = [Node(n) for n in range(1, num_nodes + 1)]
        self.node_weights = node_weights
        self.bias_nodes = bias_node_nums
        self.hidden_layer_size = h

        for bias_node_num in self.bias_nodes:
            self.nodes[bias_node_num - 1].node_output = 1

        for weight in node_weights:
            i, n = weight.split(',')
            current_node = self.nodes[int(i) - 1]
            next_node = self.nodes[int(n) - 1]
            next_node.parents.append(current_node)

    def build_neural_net(self, input_array):
        for n in range(0, len(input_array)):
            self.nodes[n].node_input = input_array[n]
            self.nodes[n].node_output = activation_function(input_array[n])

        for node in self.nodes:
            if node.node_num in self.bias_nodes or node.node_num in [n for n in range(0, 11)]:
                continue

            total_input = 0

            for input_node in node.parents:
                total_input += input_node.node_output * self.node_weights[str(input_node.node_num) + ',' + str(node.node_num)]

            node.node_input = total_input

            node.node_output = activation_function(total_input)

        return [self.nodes[n].node_num for n in range(10 + self.hidden_layer_size + 1, len(self.nodes))]

    def get_node_outputs(self):
        info_dict = {}

        for node in self.nodes:
            info_dict[node.node_num] = node.node_output

        return info_dict

    def get_node(self, node_num):
        return self.nodes[node_num - 1]


def get_weight_ids(layer_sizes, bias_node_nums):
    weight_ids = []
    num_nodes_in_layer = {}
    num_nodes = 0
    nodes = {}

    for n in range(1, len(layer_sizes) + 1):
        num_nodes_in_layer[n] = []
        nodes[n] = []

    for key in num_nodes_in_layer:
        max_node_num_in_layer = layer_sizes[key - 1]

        if key == 1 or key == len(layer_sizes):
            num_nodes_in_layer[key] = [n for n in range(1, layer_sizes[key - 1] + 1)]

        else:
            num_nodes_in_layer[key] = [n for n in range(1, layer_sizes[key - 1] + 1)]

    for key in num_nodes_in_layer:
            
        for node_num in num_nodes_in_layer[key]:
            num_nodes += 1
            nodes[key].append(str(num_nodes))

    
    for key in range(1, len(nodes.keys())):
        for n in nodes[key]:

            for num in nodes[key + 1]:
                if int(num) not in bias_node_nums:
                    weight_ids.append(n + ',' + num)

    return weight_ids


def make_new_gen_v2(parents):
    new_gen = parents.copy()

    for parent in parents:
        child_weights = {}

        if random.randint(0, 1) == 0:
            add_subtract = random.choice(["Add", "Delete"])

            if add_subtract == "Add" and child_hidden_layer_size < 10:
                child_weight_ids = get_weight_ids([10, parent.hidden_layer_size + 1 + 1, 9], [10, 10 + parent.hidden_layer_size + 1 + 1])
                parent_weight_ids = list(parent.node_weights.keys())

                for weight in child_weight_ids:
                    if weight in parent_weight_ids:
                        child_weights[weight] = parent.node_weights[weight] + np.random.normal(0, 0.05)

            else:
                child_weights[weight] = 0

            if add_subtract == "Subtract" and child_hidden_layer_size > 1:
                selected_node = random.choice([parent.nodes[n] for n in range(11, 11 + parent.hidden_layer_size + 1)])
                parent.nodes.remove(parent.nodes[selected_node - 1])


        child = EvolvedNeuralNet(child_num_nodes, child_weights, [10, 10 + child_hidden_layer_size + 1], child_hidden_layer_size)
        new_gen.append(child)

    return new_gen

def make_new_gen(parents):
    new_gen = parents.copy()

    for parent in parents:
        child_weights = {}
        child_num_nodes = len(parent.nodes)
        child_hidden_layer_size = parent.hidden_layer_size

        if random.randint(0, 1) == 0:
            add_subtract = random.choice(["Add", "Delete"])

            if add_subtract == "Add" and child_hidden_layer_size < 10:
                child_num_nodes += 1
                child_hidden_layer_size += 1

            if add_subtract == "Subtract" and child_hidden_layer_size > 1:
                child_num_nodes -= 1
                child_hidden_layer_size -= 1

        child_weight_ids = get_weight_ids([10, child_hidden_layer_size + 1, 9], [10, 10 + child_hidden_layer_size + 1])
        parent_weight_ids = list(parent.node_weights.keys())
#make randomly selected node deleted from parent
        for weight in child_weight_ids:
            if weight in parent_weight_ids:
                child_weights[weight] = parent.node_weights[weight] + np.random.normal(0, 0.05)

            else:
                child_weights[weight] = 0

        child = EvolvedNeuralNet(child_num_nodes, child_weights, [10, 10 + child_hidden_layer_size + 1], child_hidden_layer_size)
        new_gen.append(child)

    return new_gen


def run_game(players):
    game = TicTacToe(players)
    game.run_to_completion()


def make_first_gen(population_size):
    first_gen = []

    for n in range(population_size):
        h = random.randint(1, 10)
        weight_ids = get_weight_ids([10, h + 1, 9], [10, 10 + h + 1])
        weights = {}

        for weight_id in weight_ids:
            weights[weight_id] = random.uniform(-0.5, 0.5)	

        neural_net = EvolvedNeuralNet(10 + h + 1 + 9, weights, [10, 10 + h + 1], h)
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
    return sorted_nets[:15]

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

def find_max_total_payoff_from_first_evaluation_data(first_evaluation_data):
    max_total_payoff_net = list(first_evaluation_data.keys())[0]

    for neural_net in first_evaluation_data:
        if first_evaluation_data[neural_net] > first_evaluation_data[max_total_payoff_net]:
            max_total_payoff_net = neural_net

    return first_evaluation_data[max_total_payoff_net]


def run(num_first_gen, num_gen):
    max_payoff_values = {}

    start_time = time.time()
    first_gen = make_first_gen(num_first_gen)
    first_evaluation_data = first_evaluation(first_gen)
    #print("First evaluation for Gen 0 Done")
    second_evaluation_data = second_evaluation(first_evaluation_data)
    #print("Second evaluation for Gen 0 Done")
    next_gen_parents = select_parents(second_evaluation_data)
    #print("Parents from Gen 0 have been selected")
    max_payoff_values[0] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data)
    #print("Got Max Total Payoff Value for Gen 0")
    current_gen = make_new_gen(next_gen_parents)
    print(f"Gen 0 took {time.time() - start_time} seconds to complete")


    for n in range(1, num_gen):
        start_time = time.time()
        first_evaluation_data = first_evaluation(current_gen)
        #print(f"First evaluation for Gen {n} Done")
        second_evaluation_data = second_evaluation(first_evaluation_data)
        #print(f"Second evaluation for Gen {n} Done")
        next_gen_parents = select_parents(second_evaluation_data)
        #print(f"Parents from Gen {n} have been selected")
        max_payoff_values[n] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data)
        #print(f"Got Max Total Payoff Value for Gen {n}")
        current_gen = make_new_gen(next_gen_parents)
        print(f"Gen {n} took {time.time() - start_time} seconds to complete")

    return max_payoff_values


total_values = {}

for n in range(0, 75):
    total_values[n] = 0


for n in range(0, 4):
    start_time = time.time()
    max_payoff_values = run(30, 75)

    for layer in max_payoff_values:
        total_values[layer] += max_payoff_values[layer]

    print(f"Trial {n} took {time.time() - start_time} seconds to complete")


x_values = [key for key in list(total_values.keys())]
y_values = [value / 4 for value in list(total_values.values())]

plt.style.use('bmh')
plt.plot(x_values, y_values)
plt.xlabel('num generations')
plt.ylabel('max total payoff')
plt.legend(loc="best")
plt.savefig('fogel.png')