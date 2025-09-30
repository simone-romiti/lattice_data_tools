# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.xiyey import fit_xiyey

x = np.array([[-1.0, -1.0],
              [0.0, -1.0],
              [+1.0, -1.0],
              [-1.0, 0.0],
              [0.0, 0.0],
              [+1.0, 0.0],
              [-1.0, +1.0],
              [0.0, +1.0],
              [+1.0, +1.0]])
N_pts = x.shape[0]

# Define the function y(x1, x2) = exp(-m * ((x1 - c)^2 + (x2 - c)^2))
def ansatz(x, p):
    m, c1 = p
    res = m * np.exp(-c1*(x[0] - x[1]))
    return res
#---

m_true, c1_true = 0.1, 4.0 # exact parameters
p_true = np.array([m_true, c1_true])

# Evaluate the function on the noisy grid
y_exact = np.array([ansatz(x[i,:], p_true) for i in range(N_pts)])
y = np.array([ansatz(x[i,:], p_true) for i in range(N_pts)])
ey = 0.05*np.abs(y)

guess = np.array([1.0, 1.0])

fit2_result = fit_xiyey(ansatz, x, y, ey, guess)
fit2_params = fit2_result["par"] # Extract the fitted parameters

# correlated fit
n_tot = x.flatten().shape[0] + y.shape[0]
Cov_estimate = np.mean(ey)*np.eye(n_tot)
perturbation = 0.5 * np.random.randn(n_tot, n_tot)
Cov_estimate = Cov_estimate + perturbation
Cov_estimate = Cov_estimate @ Cov_estimate.T
fit2_corr_result = fit_xiyey(ansatz, x, y, ey, guess, Cov_estimate=Cov_estimate)
fit2_corr_params = fit2_corr_result["par"] # Extract the fitted parameters



# Print the true and fitted parameters
print("===============")
print("True parameters:")
print("m_true =", m_true)
print("c1_true =", c1_true)
# print("c2_true =", c2_true)

print("=================")
print("Fitted parameters:")
print("m_fitted =", fit2_params[0], "")
print("c1_fitted =", fit2_params[1], "")
# print("c2_fitted =", fit2_params[2], "")


print("=================")
print("Fitted parameters (correlated fit):")
print("m_fitted =", fit2_corr_params[0], "")
print("c1_fitted =", fit2_corr_params[1], "")
# print("c2_fitted =", fit2_params[2], "")
