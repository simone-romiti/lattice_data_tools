"""3rd test of the bootstrap routines

Skewed distributions
"""

import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

import sys

sys.path.append('../../')

from lattice_data_tools import bootstrap_sampling as bts

np.random.seed(123)

N = 10000
mean = 0.0
sigma = 2.0
skewness = 5
print("Generating", N, "random variables with skewed distribution")

x = skewnorm.rvs(a=5, loc=mean, scale=sigma, size=N)
sinx2 = np.sin(x**2)

n_bins = int(np.sqrt(N))

pdf_values = skewnorm.pdf(x, a=skewness, loc=mean, scale=sigma)
# def weighted_average(arr):
#     return np.sum(arr * pdf_values) / np.sum(pdf_values)
# #---

sinx2_var = np.var(sinx2)
#sinx2_var = np.average(sinx2) #, weights=hist_counts)

# plt.hist(x, bins=n_bins)
# plt.savefig("histogram-original_data.pdf")
# plt.close()

N_bts = 100000
K = 2
x_bts = bts.uncorrelated_confs_to_bts(x=x, N_bts=N_bts, K=K, seed=456)
sinx2_bts = np.sin(x_bts**2)

print("================================")
print("Average and error on the average")
print("================================")
print("")
print("From the original data and the formulas from the uniform distribution:")
print(np.var(x))
# print(np.sqrt(sinx2_var/K))
print("From the bootstrap data:")
print((x_bts**2 - (x_bts.mean())**2).average())
# print(sinx2_bts.std())
# print("dx_avg:", np.std(x_bts))
