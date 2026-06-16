"""
Checking the sampling for a single link system in SU(2).
This is equivalent to a single-plaquette system, as 3 of the 3 gauge links can be gauged to 1 in the path integral.

This script can test the sampling from:
- |Tr(U)|^2
- The uniform distribution over the Haar measure

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
d = 1
L = 1
L_mu = d*[L]
Nc = 2
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260610

U0 = GaugeConfiguration.from_hotstart(
    batchsize=200, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

# U0 = GaugeConfiguration.from_coldstart(
#     batchsize=1, L_mu=L_mu, Nc=Nc,
#     dtype=torch.complex128, device=device,
#     requires_grad=True)


# strong coupling eigenfunction

# use this for testing the uniform distribution
# psi_SC = lambda Ui: torch.ones_like(suN.get_Tr(Ui).as_subclass(torch.Tensor)).squeeze()

# plaquette squared
# use this to test the sampling from the simplest eigenstate in the strong coupling limit
psi_SC = lambda Ui: (suN.get_Tr(Ui)).as_subclass(torch.Tensor).squeeze()/Nc
abs_psi_SC = lambda Ui: psi_SC(Ui).abs()
abs_psi2_SC = lambda Ui: abs_psi_SC(Ui)**2
log_abs_psi2_SC = lambda Ui: 0.0*torch.log(abs_psi2_SC(Ui))+0.0*1j


# Langevin dynamics

eps = 0.01
Nt0 = 1000

LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
langevin_evolution = LG.evolve(
    U=U0, eps=eps, N=Nt0, seed=seed,
    omeas = lambda i, Ui: None #print(i, Ui.average_ReTr_plaquette().mean().item())
)
U_batch = LG.generate_batch(U_initial=langevin_evolution["U"], eps=eps, N=Nt0, seed=seed)

print(U_batch.shape)

print("Generated batch of configurations:", U_batch.shape)
psi2_values= abs_psi2_SC(U_batch)

avg_plaq = suN.get_Tr(U0).as_subclass(torch.Tensor).squeeze()/Nc
omega = Nc*torch.arccos(avg_plaq.real)
y = (1/np.pi)*torch.sin(omega/2)**2 * abs_psi2_SC(U0)
plt.scatter(omega.detach().numpy(), y.detach().numpy(), color="orange")


N_bins = int(np.sqrt(omega.numel()))

avg_plaq_evol = suN.get_Tr(U_batch).as_subclass(torch.Tensor)/Nc
omega_evol = 2*torch.arccos(avg_plaq_evol).real.squeeze()
omega_sorting = torch.sort(omega_evol)
omega_evol = omega_evol[omega_sorting[1]]
y_evol = torch.sin(omega_evol/2)**2 * (abs_psi2_SC(U_batch)[omega_sorting[1]])
print(omega_evol.shape, avg_plaq_evol.shape)
print(torch.trapezoid(y=y_evol, x=omega_evol))
y_evol_norm = y_evol / torch.trapezoid(y=y_evol, x=omega_evol)


plt.scatter(omega_evol.detach().numpy(), y_evol_norm.detach().numpy(), color="green")
plt.hist(omega_evol, density=True, alpha=0.2, bins=N_bins)
plt.show()



N_epochs = 500
