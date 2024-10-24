# Load necessary libraries

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../')
from lattice_data_tools.artifacts import lambda_method

N_pts = 4

# Create an array of 10 points for "a"
a = np.linspace(0.001, 0.01, N_pts)

# Calculate the difference between y1 and y2
diff = -10.0*(a**2) + 4.0*a/np.log(a) -  a**2 - 2.0*a**4

# Generate y1 and y2 arrays
y1 = 0.02 + 25.0 * a**2 - 5.0*a**4
y2 = y1 + diff

def ansatz(ai, p):
    return p[0] + p[1]*(ai**2) + p[2]*(ai**4) + p[3]*ai/np.log(ai)
#---

guess = np.array([0.0, 0.0, 0.0, 0.0])
L0 = lambda_method(a)
mini1 = L0.y1y2_lambda_fixed(y1, y2)
mini2 = L0.y1y2_lambda_variable(y1, y2, ansatz=ansatz, guess=guess)
y_lambda1 = mini1["y"]
y_lambda2 = mini2["y"]

print("1: lambda", mini1["lambda"])
print("1: residue", mini1["residue"])
print("2: lambda", mini2["lambda"])
print("2: residue", mini2["residue"])

plt.plot(a, y1, label="1", linestyle="None", marker="x")
plt.plot(a, y2, label="2", linestyle="None", marker="x")
plt.plot(a, y_lambda1, label="y(lambda)", linestyle="None", marker="x")
plt.plot(a, y_lambda2, label="y(lambda(a))", linestyle="None", marker="o")

plt.legend()
#plt.show()
plt.savefig("./lambda_method.pdf")


