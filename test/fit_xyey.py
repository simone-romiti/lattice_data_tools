# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.legacy.xyey import fit_xyey as old_fit_xyey
from lattice_data_tools.fit.xyey import fit_xyey


N_pts = 100
x = np.linspace(0, 1, N_pts)
Nx = x.flatten().shape[0] # ==N_pts

# True y values from the function y(x) = sin(x)
omega_exact = 3.0
def ansatz(x, params):
    omega = params[0]
    return np.tanh(omega*x)
####
y_exact = ansatz(x, [omega_exact])

# Add some random noise to the y values to simulate measurement errors
np.random.seed(1243)
noise_fact = 0.01
noise = np.random.normal(0, noise_fact, size=x.shape)
y = y_exact + noise
ey = noise


guess = np.array([0.0])
fit1 = old_fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)


fit2 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)

# correlated fit
n_tot = x.flatten().shape[0] + y.shape[0]
Cov_y_inv = np.diag(1/ey**2) # just for testing the routine, it actually should produce the same results of the uncorrelated fit above

fit2_corr = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess, Cov_y_inv=Cov_y_inv)

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

