from random import random
import math

class RandomPlayer():
    def __init__(self, strategy):
        self.player_num = None
        self.strategy = strategy

    def set_player_numbers(self, n):
        self.player_num = n

    def board_index_to_move(self, board_index):
        if board_index < 3:
            desired_row = 0
            desired_column = board_index

        elif board_index < 6:
            desired_row = 1
            desired_column = board_index - 3

        elif board_index < 9:
            desired_row = 2
            desired_column = board_index - 6

        return (desired_row, desired_column)

    def choose_move(self, game_state_data):
        chosen_index = None

        for state in self.strategy:
            if state == self.get_game_state(game_state_data):
                chosen_index = self.strategy[state]

        self.board_index_to_move(chosen_index)

        return self.board_index_to_move(chosen_index)

    def get_game_state(self, game_state_data):
        game_state = ''

        for player_num in game_state_data:
            if player_num == 0:
                game_state += str(player_num)

            if player_num != self.player_num:
                game_state += str(player_num)

            if player_num == self.player_num:
                game_state += str(player_num)

        return game_state
