# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.xiexiyey import fit_xiexiyey
# from lattice_data_tools.fit.trajectory import fit_xiexiyey as traj_fit_xiexiyey

x = np.array([[-1.0, -1.0],
              [0.5, -1.0],
              [+1.0, -1.0],
              [-1.0, 0.5],
              [0.5, 0.5],
              [+1.0, 0.5],
              [-1.0, +1.0],
              [0.5, +1.0],
              [+1.0, +1.0]])
N_pts = x.shape[0]
ex = 0.05*np.abs(x)

# Define the function y(x1, x2) = exp(-m * ((x1 - c)^2 + (x2 - c)^2))
def ansatz(x, p):
    m, c1, c2 = p
    res = m * ((x[0] - c1)**2 + (x[1] - c2)**2)**(13/7)
    return res
#---

m_true, c1_true, c2_true = 0.5, 4.0, 3.0 # exact parameters
p_true = np.array([m_true, c1_true, c2_true])

# Evaluate the function on the noisy grid
y_exact = np.array([ansatz(x[i,:], p_true) for i in range(N_pts)])
y = np.array([ansatz(x[i,:], p_true) for i in range(N_pts)])
ey = 0.05*np.abs(y)

guess = np.array([0.8, 5.0, 2.0])

# Call your fit_xiexiyey function to fit the model to the data
print("Uncorrelated fit")
fit1_result = fit_xiexiyey(ansatz, x, ex, y, ey, guess)
fit1_params = fit1_result["par"]

# correlated fit
n_tot = x.flatten().shape[0] + y.shape[0]
# just for testing the routine, it actually should produce the same results of the uncorrelated fit above
Cov_inv = np.nan_to_num(np.diag(1/np.concatenate((ex.flatten(), ey))**2), nan=0.0) 

# add small random noise
fit2_result = fit_xiexiyey(ansatz, x, ex, y, ey, guess, Cov_inv=Cov_inv)
fit2_params = fit2_result["par"]


# Print the true and fitted parameters
print("===============")
print("True parameters:")
print("m_true =", m_true)
print("c1_true =", c1_true)
print("c2_true =", c2_true)

print("=================")
print("Fitted parameters:")
print("m_fitted =", fit1_params[0], "")
print("c1_fitted =", fit1_params[1], "")
print("c2_fitted =", fit1_params[2], "")

print("=================")
print("Fitted parameters:")
print("m_fitted =", fit2_params[0], "")
print("c1_fitted =", fit2_params[1], "")
print("c2_fitted =", fit2_params[2], "")

