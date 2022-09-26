import random
import time
import numpy as np
import itertools

s = '0001'
nums = list(s)
perms = list(itertools.permutations(list(str(0)*3 + str(1))))
print([''.join(perms) for perms in perms])

