"""
Machine Learning a strong coupling eigenstate using Langevin dynamics to check the normalization
"""


import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.langevin import LangevinDynamics


from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

def perf(fun, info: str):
    t1 = time.time()
    res = fun()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res


print("===============================")
print("L-CNN + MLP implementation test")
print("===============================")

device = torch.device("cpu")
B = 1
d = 2
L = 2
L_mu = d*[L]
Nc = 2
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260610

U0 = GaugeConfiguration.from_hotstart(
    batchsize=50, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

# U0 = GaugeConfiguration.from_coldstart(
#     batchsize=1, L_mu=L_mu, Nc=Nc,
#     dtype=torch.complex128, device=device,
#     requires_grad=True)

# psi model
K = L//2
LCNN_layer = LCNN(U=U0, K=K)
W = LCNN_layer.get_W(U=U0)

N_in = W.shape[-3]
N_out = 5

N_hidden = 2
N_neurons = [5,5]

psi = LCNN_MLP(
    U = U0,
    LCNN_layer= LCNN_layer,
    LCNN_N_in=N_in, LCNN_N_out = N_out,
    N_hidden = N_hidden, N_neurons = N_neurons,
    seed = seed,
    act_fun_MLP = torch.nn.Tanh()
    )


# strong coupling eigenfunction
psi_SC = lambda Ui: Ui.average_Tr_plaquette()/Nc
abs_psi_SC = lambda Ui: psi_SC(Ui).abs()
abs_psi2_SC = lambda Ui: abs_psi_SC(Ui)**2
log_abs_psi2_SC = lambda Ui: torch.log(abs_psi2_SC(Ui))+0.0*1j

# Langevin dynamics

eps = 0.01
Nt0 = 1000

LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
langevin_evolution = LG.evolve(
    U=U0, eps=eps, N=Nt0, seed=seed,
    omeas = lambda i, Ui: print(i, Ui.average_ReTr_plaquette().mean().item())
)
U_batch = LG.generate_batch(U_initial=langevin_evolution["U"], eps=eps, N=Nt0, seed=seed)
print("Generated batch of configurations:", U_batch.shape)
psi2_values= abs_psi2_SC(U_batch)
#psi2_values = psi2_values[psi2_values > 1e-2]
Obs = U_batch.average_Tr_plaquette()/psi2_values
print(Obs.shape)

N_bins = int(np.sqrt(Obs.numel()))
#plt.plot(Obs)
#plt.hist(Obs) #, bins=N_bins)
#plt.show()

print(Obs.mean())

N_epochs = 500
