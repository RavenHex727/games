import copy
import time

class Node():
    def __init__(self, state, turn):
        self.state = state
        self.turn = turn
        self.winner = self.check_for_winner()
        self.previous = None
        self.children = None

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


class GameTree():
    def __init__(self, root_node):
        self.current_nodes = [Node(root_node, 1)]
        self.terminal_nodes = 0
        self.nodes_dict = {}
        self.nodes_dict[str(self.current_nodes[0].state)] = self.current_nodes[0]

    def get_free_locations(self, node):
        available_locs = []

        for row_index in range(len(node.state)):
            for column_index in range(len(node.state[0])):
                if node.state[row_index][column_index] == None:
                    available_locs.append((row_index, column_index))

        return available_locs

    def create_children(self, node):
        if node.winner != None or node.children != None:
            return

        children = []
        possible_translations = self.get_free_locations(node)

        for translation in possible_translations:
            initial_state = copy.deepcopy(node.state)
            initial_state[translation[0]][translation[1]] = node.turn

            if str(initial_state) in list(self.nodes_dict.keys()):
                children.append(self.nodes_dict[str(initial_state)])

            else:
                child = Node(initial_state, 3 - node.turn)
                self.nodes_dict[str(child.state)] = child
                child.previous = node
                children.append(child)

        node.children = children

    def build_tree(self):
        if len(self.current_nodes) == 0:
            return

        children = []

        for node in self.current_nodes:
            self.create_children(node)

            if node.children != None:
                for child in node.children:
                    if child not in children:
                        children.append(child)


            else:
                self.terminal_nodes += 1

        self.current_nodes = children
        self.build_tree()


start_time = time.time()

root_state = [[None, None, None], [None, None, None], [None, None, None]]
game = GameTree(root_state)
game.build_tree()
print(len(game.nodes_dict.keys()))
print(game.terminal_nodes)
print(time.time() - start_time)