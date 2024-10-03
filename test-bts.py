import numpy as np

import sys

sys.path.append('/home/simone/Documents/')

from lattice_data_tools import sampling

np.random.seed(123)

N=100000
x = np.random.normal(loc=0.0, scale=1.0, size=N)

N_bts = 5000
x_bts = sampling.uncorrelated_confs_to_bts(x=x, N_bts=N_bts, seed=456)

print(np.mean(x_bts), np.std(x_bts, ddof=1))


# Set the rate parameter (lambda) of the exponential distribution
lambda_param = 0.5  # For an exponential distribution, λ > 0

# Generate random points according to the exponential distribution
y = np.random.exponential(scale=1/lambda_param, size=N)
y_bts = sampling.uncorrelated_confs_to_bts(x=y, N_bts=N_bts, seed=9378)

dy_bts = np.std(y_bts, ddof=1)
dy = dy_bts*np.sqrt(N)

print(dy**2, 1/lambda_param**2)