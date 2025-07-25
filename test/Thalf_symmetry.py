

import numpy as np
# from scipy.stats import norm
import sys
import matplotlib.pyplot as plt

sys.path.append('../../')

import lattice_data_tools.symmetries as symmetries

T = 30
M = 0.8
A = 0.1

t = np.arange(0, T, 1)
C = A*np.cosh(-M*(t - T/2))
C[0] = -1 # example of contact divergence at t=0


C_symm = symmetries.impose_Thalf_symmety(C, p=1)
T_ext = C_symm.shape[0]
t_symm = np.arange(0, T_ext, 1)

plt.plot(t, C, label="C(t)")
plt.plot(t_symm, C_symm, label="C_symm(t)")

plt.show()
