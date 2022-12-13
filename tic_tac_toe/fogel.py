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
        assert len(self.nodes[1]) == 10, "Input layer has the incorrect number of nodes"
        assert len(self.nodes[3]) == 9, "Output layer has the incorrect number of nodes"
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
        for n in range(len(input_array)):
            self.nodes[1][n].node_input = input_array[n]
            self.nodes[1][n].node_output = input_array[n]

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
        nodes = [Node(n) for n in range(1, 10 + h + 1 + 9 + 1)]
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

def find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, return_net):
    max_total_payoff_net = list(first_evaluation_data.keys())[0]

    for neural_net in first_evaluation_data:
        if first_evaluation_data[neural_net] > first_evaluation_data[max_total_payoff_net]:
            max_total_payoff_net = neural_net

    if return_net == True:
        to_print_data = copy.deepcopy(max_total_payoff_net.__dict__)
        to_print_data['nodes'] = {1: [node.node_num for node in to_print_data['nodes'][1]], 2: [node.node_num for node in to_print_data['nodes'][2]], 3: [node.node_num for node in to_print_data['nodes'][3]]}
        file_object.write(f'{to_print_data} \n')

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
    max_payoff_values[1] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, return_net)
    #print("Got Max Total Payoff Value for Gen 0")
    current_gen = make_new_gen_v2(next_gen_parents)
    print(f"Gen 1 took {time.time() - start_time} seconds to complete")


    for n in range(2, num_gen + 1):
        start_time = time.time()
        first_evaluation_data = first_evaluation(current_gen)
        #print(f"First evaluation for Gen {n} Done")
        second_evaluation_data = second_evaluation(first_evaluation_data)
        #print(f"Second evaluation for Gen {n} Done")
        #testing next_gen_parents = select_parents(second_evaluation_data)
        next_gen_parents = select_parents(second_evaluation_data)
        #print(f"Parents from Gen {n} have been selected")

        if n == num_gen:
            max_payoff_values[n] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, True)
            file_object.write(f'{max_payoff_values} \n')

        else:
            max_payoff_values[n] = find_max_total_payoff_from_first_evaluation_data(first_evaluation_data, False)

        #print(f"Got Max Total Payoff Value for Gen {n}")
        current_gen = make_new_gen_v2(next_gen_parents)
        print(f"Gen {n} took {time.time() - start_time} seconds to complete")

    return max_payoff_values


total_values = {}
first_gen_size = 50
num_generations = 100
num_trials = 13

run(first_gen_size, num_generations)
'''
#logs.write(f'HYPERPARAMETERS \n\t Networks in first generation: {first_gen_size} \n\t Selection percentage: 0.5')

for n in range(1, num_generations + 2):
    total_values[n] = 0


for n in range(1, num_trials + 1):
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
'''


neural_net_data = {'nodes': {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2: [11, 12, 13, 14, 15, 25, 26, 27, 28, 29], 3: [16, 17, 18, 19, 20, 21, 22, 23, 24]}, 'node_weights': {'1,11': -0.020248163011887713, '1,12': -0.3769390345673617, '1,13': 0.024975398676509802, '1,14': -0.49389068812276976, '1,25': -0.17198411168883945, '1,26': -0.04987434868550322, '1,27': 0.117490542891238, '1,28': -0.062443614014461316, '1,29': 0.044989235049077675, '2,11': -0.4032854089208577, '2,12': -0.14089252611721284, '2,13': 0.003417346144859938, '2,14': 0.5921270257987291, '2,25': 0.2946201505498797, '2,26': 0.005288374374214972, '2,27': 0.036527381371276396, '2,28': 0.19260549816900763, '2,29': -0.017061597133415755, '3,11': -0.02237786932512642, '3,12': 0.057303054696721464, '3,13': -0.4378086849746097, '3,14': -0.46946472138474815, '3,25': -0.013725011673800764, '3,26': 0.20809970835545274, '3,27': -0.17830489603951377, '3,28': 0.08358233014574559, '3,29': -0.011610468272960238, '4,11': 0.40567647827994424, '4,12': -0.2581275905873721, '4,13': -0.34841356850351624, '4,14': -0.11955027596992487, '4,25': -0.022458849093165437, '4,26': 0.03024379175212416, '4,27': 0.1651337603559161, '4,28': -0.028346616205925615, '4,29': -0.025890334081475036, '5,11': 0.31240093467102975, '5,12': 0.11539991252113369, '5,13': 0.19738177860234454, '5,14': 0.6777101728757501, '5,25': -0.0431696416149966, '5,26': -0.050938026986560275, '5,27': 0.008916189406862389, '5,28': -0.09013804073099375, '5,29': 0.03169954818325937, '6,11': 0.13838256434209256, '6,12': 0.06145845688916551, '6,13': -0.5107754113787848, '6,14': -0.7709583768518702, '6,25': 0.3076289925006924, '6,26': 0.13390004121762483, '6,27': -0.09367461755814861, '6,28': -0.05723132708806497, '6,29': 0.13223947371864972, '7,11': -0.32934350936901885, '7,12': 0.18828107317181217, '7,13': 0.23804737100059292, '7,14': -0.4059351875281496, '7,25': 0.12121142710868944, '7,26': -0.10467369594629763, '7,27': 0.015826324244227086, '7,28': -0.13024525991662755, '7,29': 0.0548852571765442, '8,11': 0.21111085723045175, '8,12': -0.1623969593368256, '8,13': -0.11909474141095058, '8,14': 0.5671951485519531, '8,25': 0.12401252446930798, '8,26': 0.21134845752930478, '8,27': 0.20326566088026027, '8,28': -0.04864408601684593, '8,29': -0.04892664091161217, '9,11': 0.40559666892130897, '9,12': 0.16671310669125905, '9,13': -0.32590657646909477, '9,14': -0.8534238889547626, '9,25': -0.14335891345364496, '9,26': 0.003824661542648697, '9,27': -0.09007215884833283, '9,28': 0.03624256182206429, '9,29': 0.0023163807426183317, '10,11': -0.19957538182827736, '10,12': -0.18454887534282924, '10,13': 0.07799582084543383, '10,14': -0.34484915513834824, '10,25': -0.23414499790287224, '10,26': 0.2069545540844329, '10,27': 0.1344804138258968, '10,28': -0.03742907571673537, '10,29': -0.021731189462476402, '11,16': 0.22148466940718278, '11,17': 0.1460802370120569, '11,18': 0.1965285868471096, '11,19': -0.018376531361190246, '11,20': 0.7141786805521083, '11,21': 0.27932149176230825, '11,22': -0.4106439116969217, '11,23': -0.7983163841098169, '11,24': 0.08771062513052863, '12,16': 0.40188337873499447, '12,17': -0.026463301167812054, '12,18': 0.20760386803243497, '12,19': 0.7118997711076204, '12,20': -0.26034589754747683, '12,21': -0.28385526743333156, '12,22': -0.25167520124615983, '12,23': 0.6303591019568741, '12,24': -0.31340626608476996, '13,16': -0.3388652590562561, '13,17': -1.0655335153593577, '13,18': -0.5910038359138756, '13,19': 0.41760100874873346, '13,20': 0.3908067280059033, '13,21': 0.13542130147592102, '13,22': 0.44535126434185, '13,23': -0.21892331176220964, '13,24': 0.1704092823931144, '14,16': 0.6693996560618796, '14,17': -0.3455542607424505, '14,18': -0.6193693206836612, '14,19': -0.5853995578078128, '14,20': -0.02694159434197438, '14,21': 0.24903582344117506, '14,22': -0.5012415050889696, '14,23': -0.5791886738715594, '14,24': 0.23812884985917931, '15,16': 0.2053987583494169, '15,17': 0.06739150689049901, '15,18': -0.12099845530019734, '15,19': 0.13239577207588146, '15,20': 0.7691396118856305, '15,21': 0.10120960055444898, '15,22': -0.5177807868309522, '15,23': -0.05366889702524594, '15,24': -0.3080933725505444, '25,16': -0.147312129524182, '25,17': 0.26975186450642774, '25,18': 0.02926206693276752, '25,19': -0.11743457808319552, '25,20': -0.1718579415076945, '25,21': 0.046559937528379086, '25,22': 0.10168942929163158, '25,23': 0.1632259875931285, '25,24': 0.17675580008070227, '26,16': 0.09074696736692682, '26,17': -0.21710442399138868, '26,18': 0.16101143759708333, '26,19': -0.04592082236717629, '26,20': -0.02662539249775967, '26,21': -0.10279010085830129, '26,22': 0.003416914173836644, '26,23': 0.11946264740821819, '26,24': 0.06585770695614453, '27,16': -0.15692260383088752, '27,17': -0.17416087761616617, '27,18': -0.06112462397458806, '27,19': -0.04029731226032843, '27,20': 0.0941671559733731, '27,21': 0.10687928486121497, '27,22': 0.013329219894001869, '27,23': 0.23527789077025585, '27,24': 0.05636684029226402, '28,16': -0.0310270309061472, '28,17': 0.10967896442937264, '28,18': -0.025897002294214627, '28,19': -0.13806010506894084, '28,20': -0.009876130606718402, '28,21': -0.003030385817475989, '28,22': -0.03716052599030177, '28,23': 0.15671691899266219, '28,24': -0.2183490314180886, '29,16': 0.06586897571695054, '29,17': 0.11473111220863033, '29,18': -0.0787080757117121, '29,19': 0.00823816114760737, '29,20': 0.0118115586954209, '29,21': -0.059185060095378175, '29,22': 0.03895638293854761, '29,23': 0.08140068890542304, '29,24': 0.013306591858654543}, 'bias_nodes': [10, 15], 'hidden_layer_size': 9} 
nodes = {1: [], 2: [], 3: []}

for layer in neural_net_data['nodes']:
    for node_num in neural_net_data['nodes'][layer]:
        nodes[layer].append(Node(node_num))


neural_net = EvolvedNeuralNet(nodes, neural_net_data['node_weights'], neural_net_data['bias_nodes'], neural_net_data['hidden_layer_size'])

win_data = {1: 0, 2: 0, "Tie": 0}

for _ in range(30):
    players = [NNPlayer(neural_net), NearPerfect()]
    game = TicTacToe(players)
    game.run_to_completion()
    win_data[game.winner] += 1

print(win_data)