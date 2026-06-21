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
from lattice_data_tools.links.canonical_momenta_squared import WithAutodifferentiation as La2_autodiff

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

x0 = torch.tensor([0,1,0])

# strong coupling eigenfunction
#psi_SC = lambda Ui: Ui.plaquettes().as_subclass(torch.Tensor).squeeze()/np.sqrt(Nc)
psi_SC = lambda U: suN.get_Tr(U.plaquette(x=x0, mu=1,nu=0))
abs_psi_SC = lambda U: psi_SC(U).abs()
#alpha, beta = 1, 0
#CM = CanonicalMomenta(U=GaugeConfiguration(Ui))
abs_psi2_SC = lambda U: abs_psi_SC(U)**2
log_abs_psi2_SC = lambda U: torch.log(abs_psi2_SC(U))+0.0*1j
#log_abs_psi2_SC = lambda Ui: -Ui.Wilson_action(beta=3.0)+0.0*1j

N_evol = 100
LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
eps = 1e-2

psi_n = lambda U: suN.get_Tr(GaugeConfiguration(U).plaquette(x=x0, mu=1,nu=0))
def omeas(i, Ui):
    CM = CanonicalMomenta(U=GaugeConfiguration(Ui))
    res = [
        Ui,
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
n_therm =30
U_langevin = torch.flatten(torch.stack([omeas_i[0] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...], start_dim=0, end_dim=1)
psi_SC_tensor = torch.stack([omeas_i[1] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]
psi_n_tensor = torch.stack([omeas_i[2] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]
Oi_tensor = torch.stack([omeas_i[3] for omeas_i in LG_evol["Oi"]], dim=0)[n_therm:,...]

print(psi_SC_tensor.shape)
print(Oi_tensor.shape)

La2_psi_over_psi = torch.einsum("i...,i->i...", Oi_tensor, 1/psi_SC_tensor.squeeze())
N_tot = La2_psi_over_psi.shape[0]
La2_psi_mean = torch.tensor([(La2_psi_over_psi[i,...].abs()**2).sum(dim=-1).mean(dim=0).sum() for i in range(N_tot)])

# plt.plot(La2_psi_mean.real.squeeze().detach().numpy(), label="real")
# #plt.plot(La2_psi_mean.imag.squeeze().detach().numpy(), label="imag")
# plt.plot(psi_SC_tensor.real.squeeze().detach().numpy())
# plt.legend()
# plt.show()

N_bts = 1000
La2_psi_bts = correlated_confs_to_bts(Cg=np.array([(La2_psi_over_psi[i,...].abs()**2).sum(dim=-1).sum() for i in range(N_tot)]), N_bts=N_bts, seed=seed)


Z = ((psi_n_tensor/psi_SC_tensor).abs()**2).mean()
print("Z=", Z)
print("Numerical integral:", La2_psi_mean.mean() / Z)
print("Numerical integral (no Z):", La2_psi_mean.mean())
print("Bootstrap Numerical integral (no Z):", La2_psi_bts.mean(), La2_psi_bts.error())


print("Autodiff")
def Laf(U):
    CM = CanonicalMomenta(U=GaugeConfiguration(U))
    res = CM.La_chain_rule(f=psi_n, U=U)
    # print(res.shape)
    # quit()
    return res

CM2= CanonicalMomenta(U=GaugeConfiguration(U_langevin))
print(U_langevin.shape)
print(CM2.La_chain_rule(f=Laf, U=U_langevin))

# CM2 = La2_autodiff(U=GaugeConfiguration(U_langevin))
# La2_psi_autodiff = CM2.get_La2_per_link(f=psi_n, U=U_langevin).detach()
# print(La2_psi_autodiff.shape)
# lambda_autodiff = (La2_psi_autodiff.sum(dim=-1)*U0.n_links/psi_n(U_langevin)).mean()
# print(lambda_autodiff.real, "|", lambda_autodiff.imag)


tau_a = suN.get_generators(Nc=Nc, device=U0.device, dtype=U0.dtype)
eigenvalue = 4*(torch.einsum("aij,aji->a", tau_a, tau_a)/Nc).sum()
print("Exact:", eigenvalue.item())


