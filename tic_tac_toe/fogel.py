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
from random_player import *
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
        self.children = []


class EvolvedNeuralNet:
    def __init__(self, nodes_by_layer, node_weights, bias_node_nums, h):
        self.nodes = nodes_by_layer
        self.node_weights = node_weights
        self.bias_nodes = bias_node_nums
        self.hidden_layer_size = h

        for node in flatten(self.nodes):
            if node.node_num in self.bias_nodes:
                node.node_output = 1

        for weight in node_weights:
            i, n = weight.split(',')
            current_node = self.get_node(int(i))
            next_node = self.get_node(int(n))

            current_node.children.append(next_node)

            if int(n) not in self.bias_nodes:
                next_node.parents.append(current_node)

    def build_neural_net(self, input_array):
        for node in self.nodes[1]:
            node.node_input = input_array[n]
            node.node_output = input_array[n]

        for node in self.nodes[2] + self.nodes[3]:
            total_input = 0

            for input_node in node.parents:
                total_input += input_node.node_output * self.node_weights[str(input_node.node_num) + ',' + str(node.node_num)]

            node.node_input = total_input

            node.node_output = activation_function(total_input)

        return [node.node_output for node in self.nodes[3]]

    def get_node(self, node_num):
        for node in flatten(self.nodes):
            if node.node_num == node_num:
                return node



def flatten(input_dict):
    flattened_dict = []

    for key in input_dict:
        for value in input_dict[key]:
            flattened_dict.append(value)

    return flattened_dict


def get_weight_ids(nodes_by_layer, bias_node_nums, h):
    weight_ids = []

    for layer in nodes_by_layer:
        if layer != 3:
            for node in nodes_by_layer[layer]:
                for next_layer_node in nodes_by_layer[layer + 1]:
                    if next_layer_node.node_num not in bias_node_nums:
                        weight_ids.append(f'{node.node_num},{next_layer_node.node_num}')

    return weight_ids


def get_nodes_excluded_from_removal(nodes, bias_node_nums):
    nodes_exempt = []

    for node in nodes:
        if node.node_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            nodes_exempt.append(node)

        elif node.node_num in [node.node_num for node in nodes if len(node.children) == 0]:
            nodes_exempt.append(node)

        elif node.node_num in bias_node_nums:
            nodes_exempt.append(node)

    return nodes_exempt



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
                child_nodes[2].append(Node(max([node.node_num for node in flatten(parent.nodes)]) + 1))

            if add_subtract == "Subtract" and child_hidden_layer_size != 1:
                child_hidden_layer_size -= 1
                selected_node_num = random.choice([node.node_num for node in parent.nodes[2] if node.node_num not in parent.bias_nodes])

                for node in child_nodes[2]:
                    if node.node_num == selected_node_num:
                        child_nodes[2].remove(node)

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


def convert_to_nodes_by_layer(nodes): #special to first gen only
    nodes_by_layer = {1: [], 2: [], 3: []}

    for n in range(0, len(nodes)):
        node = nodes[n]

        if n <= 9:
            nodes_by_layer[1].append(node)

        elif node in nodes[-9:]:
            nodes_by_layer[3].append(node)

        else:
            nodes_by_layer[2].append(node)

    return nodes_by_layer

def make_first_gen(population_size):
    first_gen = []

    for n in range(population_size):
        h = random.randint(1, 10)
        nodes = [Node(n) for n in range(1, 10 + h + 1 + 9)]
        nodes_by_layer = convert_to_nodes_by_layer(nodes)
        weight_ids = get_weight_ids(nodes_by_layer, [10, 10 + h + 1], h)
        weights = {}

        for weight_id in weight_ids:
            weight = random.uniform(-0.5, 0.5)
            assert abs(weight) <= 0.5
            weights[weight_id] = weight	

        neural_net = EvolvedNeuralNet(nodes_by_layer, weights, [10, 10 + h + 1], h)
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
    return sorted_nets[:int(len(sorted_nets) / 2)]

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

    if return_net == True:
        to_print_data = copy.deepcopy(max_total_payoff_net.__dict__)
        to_print_data['nodes'] = [node.node_num for node in flatten(to_print_data['nodes'])]
        file_object.write(f'{max_total_payoff_net.__dict__} \n')

    return first_evaluation_data[max_total_payoff_net]


file_object = open('neural_nets.txt', 'a')


def run(num_first_gen, num_gen):
    max_payoff_values = {}
    return_net = False
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
            return_net = True

        max_payoff_values[n] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, return_net)

        #print(f"Got Max Total Payoff Value for Gen {n}")
        current_gen = make_new_gen_v2(next_gen_parents)
        print(f"Gen {n} took {time.time() - start_time} seconds to complete")

    return max_payoff_values

#code smth up to get neural net in last gen with highest 1st evaluation payoff
total_values = {}
first_gen_size = 10
num_generations = 20
num_trials = 2

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
plt.savefig('fogelv2.png')


'''
neural_net_data = {'node_weights': {'1,11': -0.8704943137088678, '1,12': -0.24963633794012516, '1,13': -0.2704520557660971, '1,14': 0.05418989507521883, '1,15': 0.2718208748589976, '1,16': 0.22683574691895494, '1,17': -0.8197474687712278, '1,18': -0.4126697498967016, '1,19': 0.007021663662301687, '1,20': -0.5213697536375319, '2,11': 0.09468943446752756, '2,12': 0.31750570178948356, '2,13': -0.9814770306250968, '2,14': 0.3552564317729187, '2,15': -0.3074598050054237, '2,16': 0.319869089265141, '2,17': 0.5516260110967486, '2,18': 0.44073348748523294, '2,19': -0.12532831828094024, '2,20': -0.45226305664603644, '3,11': -0.5434830807817933, '3,12': 0.3970448546419604, '3,13': 0.019755887272560765, '3,14': 0.3655250937641532, '3,15': -0.05490699033059396, '3,16': -0.08398666239816875, '3,17': -0.5800855995786373, '3,18': -0.2444980003885054, '3,19': 0.165753323004782, '3,20': -0.06747010110986928, '4,11': 0.5573916729124765, '4,12': 0.7140826660294262, '4,13': -0.572057204442121, '4,14': 0.6365945030177043, '4,15': -0.012827433361384769, '4,16': -0.18709713755143684, '4,17': -0.34094920799173145, '4,18': -0.26708482408053885, '4,19': -0.22675804891225596, '4,20': -0.5679977697552637, '5,11': 0.888312519444705, '5,12': 1.0479672256353474, '5,13': 0.3554555652132971, '5,14': -0.47483820074144223, '5,15': 0.7935516374177719, '5,16': -0.8059762774486191, '5,17': 0.1713124017668088, '5,18': -0.22897218062922195, '5,19': 0.12929978607253184, '5,20': -0.5384897824627479, '6,11': -0.20606331463072763, '6,12': -0.02430388663757635, '6,13': 0.1065891798428357, '6,14': 0.38964263243927527, '6,15': 0.0792766752591042, '6,16': -0.6615995964587038, '6,17': 0.10081204102797411, '6,18': 0.48762296564040153, '6,19': -0.28337589188950674, '6,20': -0.4813468236699679, '7,11': 0.5687684066776763, '7,12': -0.14716089281844413, '7,13': -0.0766600754491229, '7,14': 0.05319048347242923, '7,15': 0.5611225149733485, '7,16': -0.1647531609489628, '7,17': 0.008266422034861644, '7,18': -0.5526956748944246, '7,19': -0.2331227301737545, '7,20': -0.1763454659082577, '8,11': -0.3318152235261209, '8,12': 0.3087068353098388, '8,13': 0.045632092649947886, '8,14': -0.06645547543993635, '8,15': -0.12790419807484046, '8,16': 1.0226411100961446, '8,17': -0.2681249207495177, '8,18': 0.37218127125416534, '8,19': 0.6389287549444947, '8,20': 0.6218491323894426, '9,11': -0.14577010841391635, '9,12': 0.23304185596892454, '9,13': 0.02980656882686268, '9,14': 0.75457613204091, '9,15': 0.4740307167543307, '9,16': 0.036907716862331616, '9,17': 0.44872207768116523, '9,18': -0.5085537988657127, '9,19': 0.31410607202979085, '9,20': 0.7351291418646948, '10,11': -0.7076148566572409, '10,12': -0.21778126592584165, '10,13': 0.16105196170858932, '10,14': 0.34930812455570304, '10,15': 0.4051932593523659, '10,16': -0.6874389515614918, '10,17': -0.43779670375208424, '10,18': -0.200747295719272, '10,19': 0.7109479982821301, '10,20': 0.6611937755489329, '11,22': -0.42604843531441167, '11,23': -0.5573210843729458, '11,24': 0.015532202023186809, '11,25': -0.09034282917812102, '11,26': -0.728760866836858, '11,27': 0.2941744177029889, '11,28': -0.3290807976838651, '11,29': 0.06746461339076057, '12,22': 0.1132192500646234, '12,23': 0.953165468260055, '12,24': 0.501158680977528, '12,25': 0.04103521644874393, '12,26': -0.5906318696190681, '12,27': -0.07037056848035689, '12,28': -0.3301546250776092, '12,29': 0.9623982040553022, '13,22': -0.19417688811612893, '13,23': -0.2829034025852398, '13,24': -0.20504370662776497, '13,25': 0.33101718053769796, '13,26': -0.06684057607642035, '13,27': -0.6737598353048977, '13,28': 0.44073920923476284, '13,29': -0.33849790115956996, '14,22': 0.25068238180015484, '14,23': 0.09311388433365703, '14,24': 0.08290243476207179, '14,25': -0.16070137307905924, '14,26': 0.46876815517297815, '14,27': -0.4898882482995792, '14,28': -0.24204458669348308, '14,29': -0.4386137598657349, '15,22': -0.7527049811624907, '15,23': -0.5171528250860665, '15,24': -1.0994233591471219, '15,25': 0.2564988457397281, '15,26': 0.04063463337711652, '15,27': -0.41167991565664347, '15,28': -0.8348657105729164, '15,29': -0.03840606255068234, '16,22': -0.14063027665487954, '16,23': -0.35387109970007974, '16,24': 0.11161959355255266, '16,25': -0.49176675469345177, '16,26': 0.43797972441321564, '16,27': -0.12517822465949457, '16,28': 0.11875047935697619, '16,29': 0.09451321267730262, '17,22': -0.19864114903875268, '17,23': 0.047710518936547694, '17,24': 0.5921535391271515, '17,25': 0.027705384201204303, '17,26': 0.8115877344636612, '17,27': -0.24836608238702138, '17,28': 0.3517839676617872, '17,29': -0.6608550909700038, '18,22': -0.6025104606258868, '18,23': 0.5391219955871818, '18,24': -0.2522136473905843, '18,25': 0.29775980709259453, '18,26': 1.0689706162064254, '18,27': 0.21977456372693302, '18,28': 0.4458932838660097, '18,29': -0.10206338817647127, '19,22': -0.002120510800100381, '19,23': 1.3042630425126818, '19,24': -0.2490726707588289, '19,25': -0.29688277132071855, '19,26': -0.36047375146794575, '19,27': -0.34897030558730413, '19,28': -0.06492029645442791, '19,29': -0.047256898887599, '20,22': 0.326383535212714, '20,23': 0.8461513615894244, '20,24': -0.8703065338768038, '20,25': 0.3995090851681349, '20,26': -0.2085269297806196, '20,27': -0.6463972616686627, '20,28': 0.7987484489384089, '20,29': -0.3448875764941071}, 'bias_nodes': [10, 21], 'hidden_layer_size': 10} 
nodes = []

for weight in neural_net_info['node_weights']:
    i, n = weight.split(',')

    if int(i) not in [node.node_num for node in nodes]:
        nodes.append(Node(int(i)))

    if int(n) not in [node.node_num for node in nodes]:
        nodes.append(Node(int(n)))


sorted_nodes = sorted(nodes, key=lambda x: x.node_num)

neural_net = EvolvedNeuralNet(sorted_nodes, neural_net_info['node_weights'], neural_net_info['bias_nodes'], neural_net_info['hidden_layer_size'])
'''