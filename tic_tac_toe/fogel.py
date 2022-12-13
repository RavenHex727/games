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
'''
run(first_gen_size, num_generations)

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


neural_net_data = {'nodes': {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2: [11, 12, 13, 14, 15, 25, 26, 27, 28, 29, 30], 3: [16, 17, 18, 19, 20, 21, 22, 23, 24]}, 'node_weights': {'1,11': -0.0787450943360702, '1,12': 0.11628286547565009, '1,13': 0.2020566790460185, '1,14': -0.25525529660508806, '1,25': -0.5591731185106588, '1,26': 0.29490261003159324, '1,27': -0.10309655779729723, '1,28': 0.10565093104229448, '1,29': 0.016076425244713484, '1,30': -0.147013117326152, '2,11': 0.2732546569297076, '2,12': -0.8150550199271556, '2,13': 0.8323956556036014, '2,14': -0.5854256946224526, '2,25': -0.0373956522799863, '2,26': 0.15953240943308772, '2,27': -0.36763869841377095, '2,28': -0.18337945238767495, '2,29': 0.12183197884277445, '2,30': -0.09156002939231803, '3,11': -0.6411331711268249, '3,12': -0.45414952360941885, '3,13': 0.4198713464078764, '3,14': 0.3076600310505822, '3,25': 0.04365907501299532, '3,26': 0.0392633733930852, '3,27': -0.16868744804402297, '3,28': 0.18308203019993846, '3,29': -0.021648002887694577, '3,30': 0.06221752917239327, '4,11': 1.057975731216833, '4,12': 0.12581429259532334, '4,13': 0.06266434243129942, '4,14': 0.5231674704860481, '4,25': -0.4123811583364098, '4,26': 0.29178179952667427, '4,27': 0.10662911151918926, '4,28': -0.0273552617156528, '4,29': 0.03476505952105906, '4,30': -0.061097929493074694, '5,11': 0.6326720204976278, '5,12': 0.008836286635166106, '5,13': 0.20882799026575544, '5,14': -0.39139133703818585, '5,25': -0.3467476808543343, '5,26': 0.21579983255644453, '5,27': 0.1671965561831652, '5,28': -0.23556163941171737, '5,29': 0.061408290112497696, '5,30': 0.14328094798364865, '6,11': -0.13663326109593968, '6,12': 0.09334434740311232, '6,13': -0.25652824892524034, '6,14': -0.3860551365250179, '6,25': 0.14380377873701183, '6,26': 0.15131550566822044, '6,27': -0.04936999711116706, '6,28': -0.13636626414531383, '6,29': -0.042705488262330164, '6,30': -0.001051110132431296, '7,11': 0.7537138692142941, '7,12': -0.24426385279907012, '7,13': 0.5689272093586165, '7,14': -0.3472161830826231, '7,25': -0.24766431913303869, '7,26': -0.3513847323700182, '7,27': -0.09191040362457376, '7,28': 0.5948909655601451, '7,29': -0.08458718987000727, '7,30': 0.016291579027282628, '8,11': 0.002734805585383304, '8,12': 0.03604323927742124, '8,13': -0.18930848212555723, '8,14': 0.42269164940269416, '8,25': -0.1735081758915321, '8,26': 0.23564653869456326, '8,27': -0.12155270646837925, '8,28': -0.081967286619301, '8,29': 0.2551992278465294, '8,30': 0.013012658421254566, '9,11': -0.3521955503907777, '9,12': -0.19758307479830178, '9,13': -0.1121628101107286, '9,14': 0.11000322746576038, '9,25': -0.10094064647611496, '9,26': -0.29739569362554086, '9,27': -0.25749980460397004, '9,28': 0.029489529404815897, '9,29': -0.1062003824849143, '9,30': 0.041817523696141765, '10,11': 0.32484457526959454, '10,12': -0.3613825121040005, '10,13': 0.32541313996847737, '10,14': -0.22174781046991418, '10,25': -0.26020423104844953, '10,26': 0.041513612262737976, '10,27': 0.5166054899251271, '10,28': -0.08999859103335703, '10,29': -0.18950427298061795, '10,30': -0.1163961686763071, '11,16': -0.07284884059190155, '11,17': -0.15247265217537487, '11,18': 0.6144830485385504, '11,19': -0.39259670687049214, '11,20': 0.17017700942463304, '11,21': -0.5750614599768231, '11,22': -0.40678879948654706, '11,23': -0.2886201475239526, '11,24': 0.5631439077160438, '12,16': -0.6176661233596874, '12,17': 0.522227669893039, '12,18': 0.01915451490891598, '12,19': -0.17123541408390996, '12,20': -0.1888120089288756, '12,21': -0.4195127244293962, '12,22': -0.3369063916200181, '12,23': 0.0535204052506901, '12,24': 0.41125516276459917, '13,16': 0.3901736345544082, '13,17': -1.037267118591345, '13,18': -0.6749988729461922, '13,19': -0.7557489346326697, '13,20': 0.2810388093844799, '13,21': -0.1747989714614179, '13,22': 0.589160457031796, '13,23': 0.08198406786146614, '13,24': -0.18486332428667362, '14,16': -0.7520728765461103, '14,17': -0.10372516139332166, '14,18': -0.4983502712723137, '14,19': 0.2234676811564933, '14,20': 0.45342428693237136, '14,21': -0.03789280871341687, '14,22': -0.048253112344445626, '14,23': -0.06746208609475521, '14,24': 0.392078328080575, '15,16': -0.6619960929265405, '15,17': 0.4781033961905825, '15,18': 0.328642469462615, '15,19': 0.5803937980500694, '15,20': 0.5827228954086421, '15,21': 0.11613966859504227, '15,22': 0.16370212421162716, '15,23': 0.3325910721871727, '15,24': 0.10169702769634215, '25,16': 0.26967097546348223, '25,17': -0.4512099803645857, '25,18': 0.6375065457002316, '25,19': 0.23137894058246486, '25,20': 0.11557225581030951, '25,21': -0.277414263006273, '25,22': 0.17671425276694985, '25,23': -0.03199750375370925, '25,24': 0.17595930510197244, '26,16': -0.10759871145542309, '26,17': 0.1213873558602623, '26,18': 0.28085909311222307, '26,19': 0.1854609876277044, '26,20': 0.21580414247368662, '26,21': -0.22347513210438114, '26,22': -0.2840935434288531, '26,23': 0.2128136798547215, '26,24': 0.11613094284589855, '27,16': -0.04997137112456398, '27,17': 0.1288888764006536, '27,18': -0.02595558254562809, '27,19': -0.40563531039440365, '27,20': 0.3874997730899038, '27,21': 0.037688720862025286, '27,22': -0.06566447840260106, '27,23': 0.010951304470499357, '27,24': 0.271651554282061, '28,16': 0.18056844680139283, '28,17': -0.15840559196745416, '28,18': -0.3004907526171216, '28,19': 0.10645111655511738, '28,20': 0.288995920242009, '28,21': -0.1868011481567733, '28,22': -0.0695659585624139, '28,23': -0.10852657115168206, '28,24': 0.15212152161529938, '29,16': -0.108283532376781, '29,17': -0.011157635663551909, '29,18': 0.30119967795367614, '29,19': 0.04784735481727401, '29,20': -0.3653991097422852, '29,21': 0.05937961756975807, '29,22': -0.15958061859366213, '29,23': -0.10815801470714129, '29,24': 0.09925575461160029, '30,16': 0.00030559291394901406, '30,17': 0.009824792438608943, '30,18': 0.054476517887052184, '30,19': 0.09703504667698536, '30,20': -0.0749337943875686, '30,21': -0.02385893667831459, '30,22': 0.14169168320304185, '30,23': 0.005613496824690076, '30,24': -0.08930688965838843}, 'bias_nodes': [10, 15], 'hidden_layer_size': 10} 
nodes = {1: [], 2: [], 3: []}

for layer in neural_net_data['nodes']:
    for node_num in neural_net_data['nodes'][layer]:
        nodes[layer].append(Node(node_num))

neural_net = EvolvedNeuralNet(nodes, neural_net_data['node_weights'], neural_net_data['bias_nodes'], neural_net_data['hidden_layer_size'])
win_data = {1: 0, 2: 0, "Tie": 0}

for _ in range(100):
    players = [NNPlayer(neural_net), NearPerfect()]
    game = TicTacToe(players)
    game.run_to_completion()
    win_data[game.winner] += 1

print(win_data)