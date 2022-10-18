import random
import time
import numpy as np
import itertools

def transpose(input_board):
    return np.array(input_board).T.tolist()

print(transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))