"""
Parametrix bootstrap samples from the means, errors and correlation among the variables.

If one is given the estimates of the mean values, their errors and how the variables are correlated among each other (i.e. the correlation matrix), it is possible to generate parametric bootstrap samples that are distributed according to those means, errors and that correlation.

This program tests the routine that achieves that. 

"""
import numpy as np

import sys
sys.path.append('../../')
from lattice_data_tools.bootstrap import ParametricBootstraps

RNG_seed=123 # random-number-generator seed


# exact correlation matrix (it has to be positive definite)
rho_exact = np.array([
    [1.0, 0.82, 0.48],
    [0.82, 1.0, 0.33],
    [0.48, 0.33, 1.0]
])


x_mean = np.array([10.0, 20.0, 30.0]) # mean values of the variables x_i
ex =     np.array([3.0, 6.0, 9.0]) # errors on the x_i
N_bts = 1000 # number of bootstraps

# bootstrap samples of the observables x_i, correlated according to rho
x_bts = ParametricBootstraps.from_x_ex_rho(x=x_mean, dx=ex, rho=rho_exact, N_bts=N_bts, seed=RNG_seed)

rho_estimated = x_bts.correlation_matrix() # correlation matrix estimated from the generated bootstraps


print("Exact correlation matrix:")
print(rho_exact)

print("Correlation matrix estimated from the generated samples")
print(rho_estimated)

print("Increase N_bts in the code in order to see the convergence of rho_estimated to rho_exact")

