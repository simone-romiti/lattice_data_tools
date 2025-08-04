""" Simple testing of Nested Sampling (NS) routines on some syntetic data """

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../')
from lattice_data_tools.nested_sampling.phase_space import get_log_t, get_log_X, get_log_w
from lattice_data_tools.nested_sampling.weights import get_log_wL_unnormalized, get_log_wL_normalized, get_log_Z_curve, get_idx_pruned_interval


np.random.seed(192837465)

# configuration for a lattice (pure gauge theory --> sampling action divided by beta)
Nt = 12 # time extent
Nx = 6 # spatial extent
volume = Nt * int(Nx**3) # total volume 
d = 4 # number of spacetime dimensions
n_plaq = d*(d-1)*volume # number of plaquettes

n_live = 10000
n_iter = 1000000 # NS iterations

beta_gen = 5.0 # value of beta used only to generate the synthetic data
n = 3
Sn = np.sort(np.random.exponential(scale = 1.0/(beta_gen), size=n_iter)) # imagine you produced this data 
S = np.power(Sn, 1/n) # like this, the distribution is: S^(n-1) * exp(-beta_gen*S^n)

# phase space variables
log_t = get_log_t(n_live=n_live, n_iter=n_iter, sampling_t=False)
log_X = get_log_X(log_t = log_t)
log_w = get_log_w(log_t=log_t, log_X=log_X, strategy="symm")

# building the partition function at a fixed beta
beta_target = 2.0
log_L = (n-1)*np.log(S) -beta_target*Sn
log_wL = get_log_wL_normalized(log_wL=get_log_wL_unnormalized(log_w=log_w, log_L=log_L))

eps_wL = 1e-8 # pruning threshold
idx_pruned = get_idx_pruned_interval(log_wL_normalized=log_wL, eps_wL=eps_wL)
log_X_pruned = log_X[idx_pruned]
log_wL_pruned = log_wL[idx_pruned]
wL_pruned = np.exp(log_wL_pruned)
S_pruned = S[idx_pruned]

fig, ax = plt.subplots(2, 2)
log_Z_pruned = get_log_Z_curve(log_wL_normalized = log_wL_pruned)


n_bins = int(np.sqrt(n_iter))
ax[0,0].hist(S, bins=n_bins, density=True)
ax[0,0].set_xlabel("$S$")
ax[0,0].set_ylabel("Density")

ax[0,1].plot(-log_X_pruned, S_pruned)
ax[0,1].set_xlabel("-$\\log{(X)}$")
ax[0,1].set_ylabel("$S$")

ax[1,0].plot(-log_X_pruned, wL_pruned)
ax[1,0].set_xlabel("-$\\log{(X)}$")
ax[1,0].set_ylabel("$w_i L_i$")

ax[1,1].plot(-log_X_pruned, np.exp(log_Z_pruned))
ax[1,1].set_xlabel("-$\\log{(X)}$")
ax[1,1].set_ylabel("$Z$")


plt.tight_layout()
plt.savefig("./nested_sampling.pdf")
# plt.show()



