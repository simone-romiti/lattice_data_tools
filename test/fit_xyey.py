# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.legacy.xyey import fit_xyey as old_fit_xyey
from lattice_data_tools.fit.xyey import fit_xyey


N_pts = 100
x = np.linspace(0, 1, N_pts)

# True y values from the function y(x) = sin(x)
omega_exact = 3.0
y_exact = np.exp(omega_exact*x)

# Add some random noise to the y values to simulate measurement errors
np.random.seed(283754)
noise = np.random.normal(0, 0.1, size=x.shape)
y = y_exact + noise
ey = 0.3*np.abs(y)

def ansatz(x, params):
    omega = params[0]
    return np.exp(omega*x)
####

guess = np.array([0.0])
fit1 = old_fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)


fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)

# correlated fit
n_tot = x.flatten().shape[0] + y.shape[0]
Cov_estimate = np.eye(n_tot)
perturbation = 0.05 * np.random.randn(n_tot, n_tot)
Cov_estimate = Cov_estimate + perturbation
Cov_estimate = Cov_estimate @ Cov_estimate.T

fit2_corr = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess, Cov_estimate=Cov_estimate)

print("Compare the 2 fit routines (should give the same result)")
for fit in [fit1, fit2, fit2_corr]:
    print("---")
    for k in ["par", "ch2"]:
        print(k, fit[k])


print("==============")
print("Exact results:")
print("omega_exact (par):", omega_exact)

plt.plot(x,y)
plt.plot(x, y_exact)
plt.savefig("./fit_xyey.pdf")

