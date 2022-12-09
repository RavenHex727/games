import random
import time
import numpy as np
import itertools

def transpose(input_board):
    return np.array(input_board).T.tolist()

game_board = [[1,0,1], [2,0,1], [0,0,0]]
print([1] + [2])