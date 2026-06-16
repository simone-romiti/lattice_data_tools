"""
Generating samples accoding to the |\\psi|^2 of a strong coupling eigenstate and measuring the energy.
"""

from tqdm import tqdm
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
from lattice_data_tools.bootstrap import BootstrapSamples, uncorrelated_confs_to_bts, correlated_confs_to_bts
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta

from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

device = torch.device("cpu")
B = 1
d = 3
L = 3
L_mu = d*[L]
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260616

U0 = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

U0 = GaugeConfiguration.from_coldstart(
    batchsize=1, L_mu=L_mu, Nc=Nc,
    dtype=torch.complex128, device=device,
    requires_grad=True)

x0 = torch.tensor([1,0,0])

# strong coupling eigenfunction
#psi_SC = lambda Ui: Ui.plaquettes().as_subclass(torch.Tensor).squeeze()/np.sqrt(Nc)
psi_SC = lambda U: 1 + 0*suN.get_Tr(U.plaquette(x=x0, mu=1,nu=0))
abs_psi_SC = lambda U: psi_SC(U).abs()
abs_psi2_SC = lambda U: abs_psi_SC(U)**2
log_abs_psi2_SC = lambda U: torch.log(abs_psi2_SC(U))+0.0*1j
#log_abs_psi2_SC = lambda Ui: -Ui.Wilson_action(beta=3.0)+0.0*1j

N_evol = 3000
LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
eps = 1e-2

psi_n = lambda U: suN.get_Tr(U.plaquette(x=x0, mu=1,nu=0))
def omeas(i, Ui):
    CM = CanonicalMomenta(U=GaugeConfiguration(Ui))
    res = [
        psi_SC(Ui),
        psi_n(Ui),
        CM.La_chain_rule(f=psi_n, U=Ui).detach()
    ]
    return res
        

LG_evol = LG.evolve(
    U=U0, eps=eps, N=N_evol, seed=seed,
    omeas = omeas
)

print(LG_evol["Oi"][0][1].shape)
n_therm =200
psi_SC_tensor = torch.stack([omeas_i[0] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]
psi_n_tensor = torch.stack([omeas_i[1] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]
Oi_tensor = torch.stack([omeas_i[2] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]

print(psi_SC_tensor.shape)
print(Oi_tensor.shape)

La2_psi_over_psi = Oi_tensor # torch.einsum("i...,i->i...", Oi_tensor, 1/psi_n_tensor.squeeze())
N_tot = La2_psi_over_psi.shape[0]
La2_psi_mean = torch.tensor([(La2_psi_over_psi[i,...].abs()**2).sum(dim=-1).mean(dim=0).sum() for i in range(N_tot)])

# plt.plot(La2_psi_mean.real.squeeze().detach().numpy(), label="real")
# #plt.plot(La2_psi_mean.imag.squeeze().detach().numpy(), label="imag")
# plt.plot(psi_SC_tensor.real.squeeze().detach().numpy())
# plt.legend()
# plt.show()

Z = (psi_n_tensor.abs()**2).mean()
print("Z=", Z)
print("Numerical:", La2_psi_mean.mean() / Z)


tau_a = suN.get_generators(Nc=Nc, device=U0.device, dtype=U0.dtype)
eigenvalue = 4*(torch.einsum("aij,aji->a", tau_a, tau_a)/Nc).sum()
print("Exact:", eigenvalue.item())
