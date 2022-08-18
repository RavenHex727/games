import copy
import time

class Node():
    def __init__(self, state, turn, player_num):
        self.state = state
        self.turn = turn
        self.player_num = player_num
        self.winner = self.check_for_winner()
        self.previous = []
        self.children = []
        self.value = None

    def get_rows(self):
        return [row for row in self.state]

    def get_columns(self):
        columns = []

        for column_index in range(len(self.state[0])):
            columns.append([row[column_index] for row in self.state])

        return columns

    def get_diagonals(self):
        diagonal1 = []
        upper_left_corner = (0, 0)
        diagonal2 = []
        upper_right_corner = (0, 2)

        for n in range(len(self.state[0])):
            diagonal1.append(self.state[upper_left_corner[0] + n][upper_left_corner[1] + n])
            diagonal2.append(self.state[upper_right_corner[0] + n][upper_right_corner[1] - n])

        return [diagonal1, diagonal2]

    def get_board_elements(self):
        board_elements = []

        for row in self.state:
            for value in row:
                board_elements.append(value)

        return board_elements

    def check_for_winner(self):
        rows_columns_diagonals = self.get_rows() + self.get_columns() + self.get_diagonals()

        for element in [element for element in rows_columns_diagonals if None not in element]:
            if len(set(element)) == 1:
                return element[0]

        if None not in self.get_board_elements():
            return "Tie"

        return None

    def children_to_value(self):
        if self.children == None or len(self.children) == 0:
            return None

        for child in self.children:
            child.set_node_value()

        return [child.value for child in self.children]

    def set_node_value(self):
        if self.children == None or len(self.children) == 0:
            if self.winner == self.player_num:
                self.value = 1

            elif self.winner == 3 - self.player_num:
                self.value = -1

            elif self.winner == 'Tie':
                self.value = 0

            return

        if self.turn == self.player_num:
            self.value = max(self.children_to_value())

        elif self.turn == 3 - self.player_num:
            self.value = min(self.children_to_value())


class GameTree():
    def __init__(self, root_state, player_num):
        self.root_node = Node(root_state, 1, player_num)
        self.current_nodes = [self.root_node]
        self.num_terminal_nodes = 0
        self.player_num = player_num
        self.nodes_dict = {str(root_state): self.root_node}

    def create_children(self, node):
        if node.winner != None or len(node.children) != 0:
            return

        children = []
        possible_translations = [(i,j) for i in range(len(node.state)) for j in range(len(node.state)) if node.state[i][j] == None]

        for translation in possible_translations:
            initial_state = copy.deepcopy(node.state)
            initial_state[translation[0]][translation[1]] = node.turn

            if str(initial_state) in list(self.nodes_dict.keys()):
                children.append(self.nodes_dict[str(initial_state)])
                self.nodes_dict[str(initial_state)].previous.append(node)
                continue

            child = Node(initial_state, 3 - node.turn, self.player_num)
            child.previous = [node]
            children.append(child)
            self.nodes_dict[str(child.state)] = child

        node.children = children

    def set_node_values(self):
        self.root_node.set_node_value()

    def build_tree(self):
        if len(self.current_nodes) == 0:
            self.current_nodes = [self.root_node]
            return

        children = []

        for node in self.current_nodes:
            self.create_children(node)

            if len(node.children) != 0:
                children += node.children

            else:
                self.num_terminal_nodes += 1

        self.current_nodes = children
        self.build_tree()

    def get_move_from_boards(self, base_state, new_state):
        base_state_children = self.nodes_dict[str(base_state)].children

        for i in range(len(new_state)):
            for j in range(len(new_state[0])):
                base = base_state[i][j]
                new = new_state[i][j]
                if base != new:
                    return (i,j)
    
    def get_best_move(self):
        base_state = self.root_node.state
        scores = [node.value for node in self.root_node.children]
        max_index = scores.index(max(scores))
        best_move_node = self.root_node.children[max_index]
        new_state = best_move_node.state
        return self.get_move_from_boards(base_state, new_state)