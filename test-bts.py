import numpy as np

import sys

sys.path.append('/home/simone/Documents/')

from lattice_data_tools import sampling

np.random.seed(123)

N=1000
x = np.random.normal(loc=0.0, scale=1.0, size=N)

N_bts = 5000
x_bts = sampling.uncorrelated_confs_to_bts(x=x, N_bts=N_bts, seed=456)

print(np.mean(x_bts), np.std(x_bts, ddof=1))