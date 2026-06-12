import numpy as np
import itertools

def get_epsilon(n):
    eps = np.zeros([n]*n, dtype=int)
    
    for perm in itertools.permutations(range(n)):
        # compute parity of permutation
        inv = 0
        for i in range(n):
            for j in range(i+1, n):
                if perm[i] > perm[j]:
                    inv += 1
        
        eps[perm] = 1 if inv % 2 == 0 else -1
    
    return eps

