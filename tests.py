import random
import time
import numpy as np
import itertools

s = '0001'
nums = list(s)
permutations = list(itertools.permutations(nums))
print([''.join(permutation) for permutation in permutations])
print(list("kingslayer, destroying castles in the sky"))

