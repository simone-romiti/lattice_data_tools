# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.xyey import fit_xyey
from lattice_data_tools.fit.xyey import polynomial_fit_xyey

N_pts = 100
x = np.linspace(-1, 1, N_pts)
Nx = x.flatten().shape[0] # ==N_pts

N_deg = 10

def ansatz(x, alpha):
    return np.poly1d(alpha[::-1])(x)
#---


np.random.seed(981)
alpha_exact = np.random.rand(N_deg+1)*2 - 1
y_exact = ansatz(x, alpha_exact)

# Add some random noise to the y values to simulate measurement errors
noise_fact = 0.05
noise = np.random.normal(0, noise_fact, size=x.shape)
y = y_exact + noise
ey = np.abs(noise)  # Use absolute value of noise as errors

guess = np.zeros_like(alpha_exact)

# non linear fit
nl_fit = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)

y_nl_fit = nl_fit["ansatz"](x, nl_fit["par"])

poly_fit = polynomial_fit_xyey(N_deg=N_deg, x=x, y=y, ey=ey)
y_poly = poly_fit["ansatz"](x, poly_fit["par"])

# print("==============")
# print("Exact results:")
# print("alpha_exact (par):", alpha_exact)

plt.errorbar(x, y_exact, yerr=ey, linestyle='None', label="data")
# plt.plot(x, y_exact, label="exact")
plt.plot(x, y_nl_fit, label="nonlinear fit")
plt.plot(x, y_poly, label="polynomial fit")
plt.legend()

output="./fit_polynomial.pdf"
print("Saving the fit plot to", output)

plt.savefig(output)

