"""2nd test of the bootstrap routines

This test shows how bootstrapping "gaussianizes" the data.

In fact, bootstrap samples are averages of a samples of data from the original dataset.
Because of the central limit theorem, bootstrap samples are distributed gaussianly
and their standard error corresponds to the error on the mean of the original distribution.

The advantage is that if we have some data whose distribution is not known analytically,
one can still estimate the mean and error on the data by bootstrapping.

NOTE: In this test we know it because we choose a uniform distribution. 
"""

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('../../')

from lattice_data_tools import sampling

np.random.seed(123)

N = 10000
a = 0.0
b = 1.0
print("Generating", N, "random variables distributed uniformly between", a, "and", b)

x = np.random.uniform(low=0.0, high=1.0, size=N)

plt.hist(x, bins=int(np.sqrt(N)))
plt.savefig("histogram-original_data.pdf")
plt.close()

N_bts = 5000
x_bts = sampling.uncorrelated_confs_to_bts(x=x, N_bts=N_bts, seed=456)

plt.hist(x_bts, bins=int(np.sqrt(N_bts)))
plt.savefig("histogram-bootstrap_data.pdf")
plt.close()

print("================================")
print("Average and error on the average")
print("================================")
print("")
print("From the original data and the formulas from the uniform distribution:")
print("x_avg:", np.average(x))
print("dx_avg:", np.sqrt((b-a)**2/12)/np.sqrt(N))
print("From the bootstrap data:")
print("x_avg:", np.average(x_bts))
print("dx_avg:", np.std(x_bts))
