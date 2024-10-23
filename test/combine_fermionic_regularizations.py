# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.combine_fermionic_regularizations import minimize_discretization_lambda_factor

N_pts = 6

# Create an array of 10 points for "a"
a = np.linspace(0.001, 0.01, N_pts)

# Calculate the difference between y1 and y2
diff = -10.0*(a**2) # 4.0*a/np.log(a) -  a**2 - 2.0*a**4

# Generate y1 and y2 arrays
y1 = 0.02 + 7.0 * a**2 # - 5.0*a**4
y2 = y1 + diff

def ansatz(ai, p):
    return p[0]*(ai**2)
#---

guess = np.array([-10000.0])
mini = minimize_discretization_lambda_factor(a, y1, y2, ansatz=ansatz, guess=guess)
y_lambda = mini["y"]

print("par", mini["par"])
print("lambda", mini["lambda"])
print("residue", mini["residue"])

# print(y_lambda)


plt.plot(a, y1, label="1", linestyle="None", marker="x")
plt.plot(a, y2, label="2", linestyle="None", marker="x")
plt.plot(a, y_lambda, label="y(lambda)", linestyle="None", marker="x")
# plt.plot((np.diff(y_lambda)/np.diff(a))**2, label="y(lambda)", linestyle="None", marker="x")

plt.legend()
plt.show()


