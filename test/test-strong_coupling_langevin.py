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
d = 1
L = 1
L_mu = d*[L]
Nc = 2
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260610

U0 = GaugeConfiguration.from_hotstart(
    batchsize=2000, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

# U0 = GaugeConfiguration.from_coldstart(
#     batchsize=1, L_mu=L_mu, Nc=Nc,
#     dtype=torch.complex128, device=device,
#     requires_grad=True)

# # psi model
# K = L//2
# LCNN_layer = LCNN(U=U0, K=K)
# W = LCNN_layer.get_W(U=U0)

# N_in = W.shape[-3]
# N_out = 5

# N_hidden = 2
# N_neurons = [5,5]

# psi = LCNN_MLP(
#     U = U0,
#     LCNN_layer= LCNN_layer,
#     LCNN_N_in=N_in, LCNN_N_out = N_out,
#     N_hidden = N_hidden, N_neurons = N_neurons,
#     seed = seed,
#     act_fun_MLP = torch.nn.Tanh()
#     )


# strong coupling eigenfunction
psi_SC = lambda Ui: (suN.get_Tr(Ui)).as_subclass(torch.Tensor).squeeze()/Nc
abs_psi_SC = lambda Ui: psi_SC(Ui).abs()
abs_psi2_SC = lambda Ui: abs_psi_SC(Ui)**2
log_abs_psi2_SC = lambda Ui: torch.log(abs_psi2_SC(Ui))+0.0*1j

# print(psi_SC(U0).mean().item(), psi_SC(U0).std().item())
# print(abs_psi2_SC(U0).mean().item(), abs_psi2_SC(U0).std().item())



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
#psi2_values = psi2_values[psi2_values > 1e-2]
# Obs = U_batch.average_Tr_plaquette()/psi2_values
# print(Obs.shape)

avg_plaq = suN.get_Tr(U0).as_subclass(torch.Tensor).squeeze()/Nc
omega = Nc*torch.arccos(avg_plaq.real)
y = (1/np.pi)*torch.sin(omega/2)**2 * abs_psi2_SC(U0)
plt.scatter(omega.detach().numpy(), y.detach().numpy(), color="orange")


N_bins = int(np.sqrt(omega.numel()))
#plt.plot(Obs)
avg_plaq_evol = suN.get_Tr(U_batch).as_subclass(torch.Tensor)/Nc
omega_evol = 2*torch.arccos(avg_plaq_evol).real.squeeze()
omega_sorting = torch.sort(omega_evol)
omega_evol = omega_evol[omega_sorting[1]]
y_evol = torch.sin(omega_evol/2)**2 * (abs_psi2_SC(U_batch)[omega_sorting[1]])
print(omega_evol.shape, avg_plaq_evol.shape)
print(torch.trapezoid(y=y_evol, x=omega_evol))
y_evol_norm = y_evol / torch.trapezoid(y=y_evol, x=omega_evol)


plt.scatter(omega_evol.detach().numpy(), y_evol_norm.detach().numpy(), color="green")
#plt.scatter(omega_evol.detach().numpy(), y_evol.detach().numpy(), color="green")
plt.hist(omega_evol, density=True, alpha=0.2, bins=N_bins)
plt.show()


#print(omega_evol)

N_epochs = 500
