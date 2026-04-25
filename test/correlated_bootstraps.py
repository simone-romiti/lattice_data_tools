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
N_bts = 10000 # number of bootstraps

print("Exact correlation matrix:")
print(rho_exact)
print("----")


# bootstrap samples of the observables x_i, correlated according to rho
for method in ["Cholesky", "eigen"]:
    print(method)
    x_bts = ParametricBootstraps.correlated_from_rho(x_mean=x_mean, x_error=ex, rho=rho_exact, N_bts=N_bts, seed=RNG_seed, method=method)
    ex_bts = x_bts.error()

    rho_estimated = x_bts.correlation_matrix() # correlation matrix estimated from the generated bootstraps
    # Cov_estimated = np.array([[rho_estimated[i,j]*ex_bts[i]*ex_bts[j]  for j in range(3)] for i in range(3)]) 

    print("Correlation matrix estimated from the generated samples")
    print(rho_estimated)

    

print("===================")

Cov_exact = np.array([[rho_exact[i,j]*ex[i]*ex[j]  for j in range(3)] for i in range(3)])

print("Exact Covariance matrix:")
print(Cov_exact)
print("---")

print("Covariance matrix estimated from the generated samples")
# bootstrap samples of the observables x_i, correlated according to rho
for method in ["Cholesky", "eigen"]:
    print(method)
    x_bts = ParametricBootstraps.correlated_from_covariance(x_mean=x_mean, x_error=ex, Cov=Cov_exact, N_bts=N_bts, seed=RNG_seed, method=method)
    ex_bts = x_bts.error()

    Cov_estimated = x_bts.covariance_matrix() # covariance matrix estimated from the generated bootstraps
    # Cov_estimated = np.array([[Cov_estimated[i,j]*ex_bts[i]*ex_bts[j]  for j in range(3)] for i in range(3)]) 

    print(Cov_estimated)

    


print("Increase N_bts in the code in order to see the convergence of rho_estimated to rho_exact")

