# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.fit.xyey import fit_xyey
from lattice_data_tools.fit.trajectory import fit_xyey as traj_fit_xyey


N_pts = 100
x = np.linspace(0, 1, N_pts)

# True y values from the function y(x) = sin(x)
omega_exact = 3
y_exact = np.exp(omega_exact*x)

# Add some random noise to the y values to simulate measurement errors
np.random.seed(283754)
noise = np.random.normal(0, 0.1, size=x.shape)
y = y_exact + noise
ey = 0.3*y

def ansatz(x, params):
    omega = params[0]
    return np.exp(omega*x)
####

guess = np.array([0.0])
fit1 = fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)


fit2 = traj_fit_xyey(ansatz=ansatz, x=x, y=y, ey=ey, guess=guess)
for fit in [fit1, fit2]:
    print("---")
    for k in ["par", "ch2"]:
        print(k, fit[k])

plt.plot(x,y)
plt.plot(x, y_exact)
# plt.show()

