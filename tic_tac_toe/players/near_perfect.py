import random
import copy
import math
import itertools

class NearPerfect:
    def __init__(self):
        self.symbol = None
        self.number = None
    
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

    def get_num_instances(self, input_list, input_string):
        num_instances = 0

        for element in input_list:
            if element in input_string:
                num_instances += 1

        return num_instances

    def check_if_list_element_in_str(self, input_list, input_string):
        for element in input_list:
            if element in input_string:
                return True

        return False
    
    def choose_move(self, game_board):
        choices = [(i,j) for i in range(len(game_board)) for j in range(len(game_board)) if game_board[i][j]==0]

        if random.randint(1, 10) == 1:
            return random.choice(choices)

        else:
            win_choices = []
            block_choices = []
            one_in_row_choices = []

            for choice in choices:
                new_board = copy.deepcopy(game_board)
                new_board[choice[0]][choice[1]] = self.number

                if self.check_for_winner(new_board) == self.number:
                    win_choices.append(choice)

                new_info = [new_board[choice[0]], [row[choice[1]] for row in new_board]]
                old_info = [game_board[choice[0]], [row[choice[1]] for row in game_board]]

                diags = [[new_board[i][i] for i in range(3)],
                        [new_board[i][2-i] for i in range(3)]]

                for n in range(0, len(diags)):
                    if diags[n] != [[game_board[i][i] for i in range(3)], [game_board[i][2-i] for i in range(3)]][n]:
                        new_info.append(diags[n])
                        old_info.append([[game_board[i][i] for i in range(3)], [game_board[i][2-i] for i in range(3)]][n])

                perms = list(itertools.permutations(list(str(3 - self.number)*2 + str(self.number))))
                perms = [''.join(perm) for perm in perms]

                perms1 = list(itertools.permutations(list(str(3 - self.number) + "0" * 2)))
                perms1 = [''.join(perm) for perm in perms1]

                for n in range(0, len(old_info)):
                    num_instances_new = self.get_num_instances(perms, "".join([str(element) for element in new_info[n]]))
                    num_instances_old = self.get_num_instances(perms, "".join([str(element) for element in old_info[n]]))

                    if num_instances_new > num_instances_old:
                        block_choices.append(choice)

                    if self.check_if_list_element_in_str(perms1, "".join([str(element) for element in old_info[n]])):
                        one_in_row_choices.append(choice)

            if len(win_choices) > 0:
                return random.choice(win_choices)

            if len(block_choices) > 0:
                return random.choice(block_choices)

            if len(one_in_row_choices) > 0:
                return random.choice(one_in_row_choices)

            return random.choice(choices)