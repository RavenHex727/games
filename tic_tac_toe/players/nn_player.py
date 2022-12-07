import random
import math
import sys

class NNPlayer():
    def __init__(self, neural_net):
        self.symbol = None
        self.number = None
        self.neural_net = neural_net
    
    def set_player_symbol(self, n):
        self.symbol = n
    
    def set_player_number(self, n):
        self.number = n

    def check_for_winner(self, board):
        rows = copy.deepcopy(board)
        cols = [[board[i][j] for i in range(3)] for j in range(3)]
        diags = [[board[i][i] for i in range(3)],
                [board[i][2-i] for i in range(3)]]

        str_info = []

        board_full = True

        for info in rows + cols + diags:
            if 0 in info:
                board_full = False

        for info in rows + cols + diags:
            for player_num in [1, 2]:
                if str(player_num) * 4 in "".join([str(element) for element in info]):
                    return player_num

        if board_full:
            return 'Tie'

        return None
    
    def get_possible_moves(self, board):
        possible_moves = [(i,j) for i in range(3) for j in range(3) if board[i][j] == 0]
        return possible_moves

    def flatten(self, game_board):
        return [game_board[i][j] for i in range(len(game_board)) for j in range(len(game_board[0]))]

    def convert_flattened_index_to_move(self, max_index):
        if max_index < 3:
            return (0, max_index)

        elif max_index < 6:
            return (1, max_index % 3)

        else:
            return (2, max_index % 3)

    def convert_to_negatives(self, game_board):
        for i in range(len(game_board)):
            for j in range(len(game_board[0])):
                if game_board[i][j] == 2:
                    game_board[i][j] = -1

        return game_board
    
    def choose_move(self, game_board):
        converted_board = self.convert_to_negatives(game_board)
        flattened_board = self.flatten(converted_board)
        assert len([n for n in flattened_board if n not in [-1, 0, 1]]) == 0, "Neural Net converted board to array incorrectly"
        results = self.neural_net.build_neural_net(flattened_board)
        assert sum(flattened_board) == 0, "Flattened board is incorrect"
        taken_spots = [n for n in range(len(flattened_board)) if flattened_board[n] != 0]

        max_index = 0

        for n in range(len(results)):
            if n not in taken_spots and results[n] > results[max_index]:
                max_index = n

        return self.convert_flattened_index_to_move(max_index)
