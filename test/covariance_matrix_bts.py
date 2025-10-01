

import numpy as np


import sys
import matplotlib.pyplot as plt

sys.path.append('../../')

from lattice_data_tools.bootstrap import BootstrapSamples, parametric_gaussian_bts

seed = 3654
N_bts = 1000
x1 = parametric_gaussian_bts(mean=0.0, error=0.1, N_bts=N_bts, seed=seed)
x2 = parametric_gaussian_bts(mean=0.0, error=0.2, N_bts=N_bts, seed=seed+10)
x = BootstrapSamples(np.vstack((x1, x2)).T)

A = np.array([[1.0, 0.8], [0.8, 1.0]])

y = BootstrapSamples.from_lambda(N_bts=N_bts, fun=lambda i: A @ x[i])

print(y.shape)

plt.plot(x[:,0], x[:,1], linestyle="None", marker=".", label="x")
plt.plot(y[:,0], y[:,1], linestyle="None", marker=".", label="y")
plt.legend()
# plt.show()


print("Uncorrelated data")
print("Correlation\n", x.correlation_matrix())
print("Covariance\n", x.covariance_matrix())

print("------------------")
print("Correlation\n", y.correlation_matrix())
print("Covariance\n", y.covariance_matrix())
