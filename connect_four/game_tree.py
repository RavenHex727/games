import copy
import time
import random
import itertools


class Node():
    def __init__(self, state, turn, player_num):
        self.state = state
        self.turn = turn
        self.player_num = player_num
        self.winner = self.check_for_winner()
        self.previous = []
        self.children = []
        self.value = None

        perms1 = list(itertools.permutations(list("000" + str(self.player_num))))
        perms2 = list(itertools.permutations(list("00" + str(self.player_num) * 2)))
        perms3 = list(itertools.permutations(list("0" + str(self.player_num) * 3)))
        perms4 = list(itertools.permutations(list("000" + str(3 - self.player_num))))
        perms5 = list(itertools.permutations(list("00" + str(3 - self.player_num) * 2)))
        perms6 = list(itertools.permutations(list("0" + str(3 - self.player_num) * 3)))
        self.perms = [perms1, perms2, perms3, perms4, perms5, perms6]

    def get_rows(self):
        return [row for row in self.state]

    def get_columns(self):
        columns = []

        for column_index in range(len(self.state[0])):
            columns.append([row[column_index] for row in self.state])

        return columns

    def get_diagonals(self):
        fdiag = [[] for _ in range(len(self.state) + len(self.state[0]) - 1)]
        bdiag = [[] for _ in range(len(fdiag))]

        for x in range(len(self.state[0])):
            for y in range(len(self.state)):
                fdiag[x + y].append(self.state[y][x])
                bdiag[x - y - (1 - len(self.state))].append(self.state[y][x])

        return fdiag + bdiag

    def get_board_elements(self):
        board_elements = []

        for row in self.state:
            for value in row:
                board_elements.append(value)

        return board_elements

    def get_columns(self):
        columns = []
        for column_index in range(len(self.state[0])):
            columns.append([row[column_index] for row in self.state])

        return columns

    def check_for_winner(self):
        rows = self.state.copy()
        cols = self.get_columns()
        diags = self.get_diagonals()

        str_info = []

        board_full = True

        for info in rows + cols + diags:
            if 0 in info:
                board_full = False

        for info in rows + cols + diags:
            if str(self.player_num) * 4 in "".join([str(element) for element in info]):
                return self.player_num

            if str(3 - self.player_num) * 4 in "".join([str(element) for element in info]):
                return 3 - self.player_num

        if board_full:
            return 'Tie'

        return None

    def check_if_list_element_in_str(self, input_list, input_string):
        for element in input_list:
            if element in input_string:
                return True

        return False

    def children_to_value(self):
        if self.children == None or len(self.children) == 0:
            return None

        for child in self.children:
            child.set_node_value()

        return [child.value for child in self.children]

    def heuristic_evaluation(self):
        rows_columns_diagonals = self.get_rows() + self.get_columns() + self.get_diagonals()

        if self.check_for_winner() != None:
            if self.check_for_winner() == self.player_num:
                return 1

            if self.check_for_winner() == 3 - self.player_num:
                return -1

            if self.check_for_winner() == "Tie":
                return 0

        else: 
            value = 0
#todo: figure out how to deal with all scenarios below except other player. multiple appearences of an item
            for element in rows_columns_diagonals: 
                if self.check_if_list_element_in_str(self.perms[2], element) and self.turn == self.player_num:
                    value += 1

                if self.check_if_list_element_in_str(self.perms[5], element) and self.turn == 3 - self.player_num:
                    value -= 1

                if self.check_if_list_element_in_str(self.perms[1], element) and self.turn == self.player_num:
                    value += 0.35

                if self.check_if_list_element_in_str(self.perms[4], element) and self.turn == 3 - self.player_num:
                    value -= 0.35

                if self.check_if_list_element_in_str(self.perms[0], element) and self.turn == self.player_num:
                    value += 0.1

                if self.check_if_list_element_in_str(self.perms[3], element) and self.turn == 3 - self.player_num:
                    value -= 0.1

            return value / 37

    def set_node_value(self):
        if self.children == None or len(self.children) == 0:
            self.value = self.heuristic_evaluation()
            return 

        if self.turn == self.player_num:
            self.value = max(self.children_to_value())

        elif self.turn == 3 - self.player_num:
            self.value = min(self.children_to_value())


class ReducedSearchGameTree():
    def __init__(self, root_state, player_num, ply):
        self.root_node = Node(root_state, 1, player_num)
        self.current_nodes = [self.root_node]
        self.num_terminal_nodes = 0
        self.player_num = player_num
        self.nodes_dict = {str(root_state): self.root_node}
        self.ply = ply

    def get_row_with_lowest_available_column(self, j, board):
        largest_row = 0

        for n in range(len(board)):
            if board[n][j] == 0:
                largest_row = n

        return largest_row

    def create_children(self, node):
        if node.winner != None or len(node.children) != 0:
            return

        children = []
        possible_translations = []

        for i in range(6):
            for j in range(7):
                if node.state[i][j] == 0 and j not in possible_translations:
                    possible_translations.append(j)

        for translation in possible_translations:
            initial_state = copy.deepcopy(node.state)
            initial_state[self.get_row_with_lowest_available_column(translation, node.state)][translation] = node.turn

            if str(initial_state) in list(self.nodes_dict.keys()):
                children.append(self.nodes_dict[str(initial_state)])
                self.nodes_dict[str(initial_state)].previous.append(node)
                continue

            child = Node(initial_state, 3 - node.turn, self.player_num)
            child.previous = [node]
            children.append(child)
            self.nodes_dict[str(child.state)] = child

        node.children = children

    def set_node_values(self, current_node):
        if current_node.value == None:
            current_node.set_node_value()

    def reset_node_values(self):
        for node in list(self.nodes_dict.values()):
            node.value = None

    def build_tree(self, current_nodes):
        children = []

        for node in current_nodes:
            self.create_children(node)

            if len(node.children) != 0:
                children += node.children

            else:
                self.num_terminal_nodes += 1

        return children

    def build_ply(self, current_node):

        self.build_tree([current_node])
        children = self.build_tree([current_node])

        for _ in range(self.ply - 1):
            self.build_tree(children)
            children = self.build_tree(children)

