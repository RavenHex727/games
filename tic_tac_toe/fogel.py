import math
import random
import itertools
import numpy as np
#import matplotlib.pyplot as plt
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


def make_new_gen(parents):
    new_gen = parents

    for parent in parents:
        child_weights = {}
        child_num_nodes = len(self.parent.nodes)
        child_hidden_layer_size = self.parent.hidden_layer_size

        if random.randint(0, 1) == 0:
            add_subtract = random.choice(["Add", "Delete"])

            if add_subtract == "Add" and child_hidden_layer_size < 10:
                child_num_nodes += 1
                child_hidden_layer_size += 1

            if add_subtract == "Subtract" and child_hidden_layer_size > 1:
                child_num_nodes -= 1
                child_hidden_layer_size -= 1

        for weight in get_weight_ids(10, child_hidden_layer_size + 1, 9):
            if weight in list(parent.node_weights.keys()):
                child_weights[weight] = parent.node_weights[weight] + np.random.normal(0, 0.05)

            else:
                child_weights[weight] = 0

        child = EvolvedNeuralNet(child_num_nodes, child_weights, parent.bias_node_nums, child_hidden_layer_size)
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



def run(num_gen):
    first_gen = make_first_gen(num_gen)
    players = [NNPlayer(first_gen[0]), NearPerfect()]
    game = TicTacToe(players)
    game.run_to_completion()
    print(game.winner)


run(1)
'''
    for neural_net in first_gen:
        neural_net.build_neural_net([0 for _ in range(9)])
'''