"""
IN PROGRESS

Machine Learning a strong coupling eigenstate using Langevin dynamics to check the normalization


- Histogram of ReTr(U) for each link
- check that the histograms look the same after Langevin


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

from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

device = torch.device("cpu")
B = 1
d = 3
L = 4
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

# strong coupling eigenfunction
#psi_SC = lambda Ui: Ui.plaquettes().as_subclass(torch.Tensor).squeeze()/np.sqrt(Nc)
psi_SC = lambda Ui: suN.get_Tr(Ui.plaquette(torch.tensor([1,0,0]), mu=1,nu=0))
abs_psi_SC = lambda Ui: psi_SC(Ui).abs()
abs_psi2_SC = lambda Ui: abs_psi_SC(Ui)**2
log_abs_psi2_SC = lambda Ui: 0.0*torch.log(abs_psi2_SC(Ui))+0.0*1j
#log_abs_psi2_SC = lambda Ui: -Ui.Wilson_action(beta=3.0)+0.0*1j

print("Thermalization")
N_evol = 10000
LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
eps = 1e-2

def omeas(i, Ui):
    return torch.stack([
        suN.get_Tr(Ui.plaquette(torch.tensor([0,1,0]), mu=0,nu=1)),
        suN.get_Tr(Ui.plaquette(torch.tensor([1,1,0]), mu=1,nu=0)),
        suN.get_Tr(Ui.plaquette(torch.tensor([0,1,2]), mu=0,nu=1)),
        suN.get_Tr(Ui.plaquette(torch.tensor([2,1,0]), mu=1,nu=0))
    ], dim=0)

thermalization = LG.evolve(
    U=U0, eps=eps, N=N_evol, seed=seed,
    omeas = omeas
)

Oi_tensor = torch.stack(thermalization["Oi"], dim=1)
print(Oi_tensor.shape)

n_therm = 1000

#Oi = abs_psi2_SC(U0)
for Oi in Oi_tensor:
    Oi2 = (Oi.view(-1).abs()**2).detach().numpy()[n_therm:]
    #print(Oi2.shape)
    # print([obs_i.abs()**2 for obs_i in thermalization["Oi"]])
    print(Oi.shape, Oi2.mean().item(), Oi.std().item()/np.sqrt(Oi.numel()))
    
    N_bts = 1000
    Oi2_bts = correlated_confs_to_bts(Cg=Oi2, N_bts=N_bts, seed=seed)
    
    #print(Oi2_bts.unbiased_mean(), Oi2_bts.error())
    #plt.title(f"Average plaquette={Oi.mean()} +/- {Oi.std()}")
    plt.plot(Oi.real, label=f"\\int |\\psi|^2 = {Oi2_bts.unbiased_mean():2e} +/- {Oi2_bts.error():2e}")

plt.title("\\psi = ReTr(P)")
plt.legend()
plt.show()
quit()

U_batch = GaugeConfiguration(thermalization["U"][0:1,...])
B_stoch = 1
for bs in range(B_stoch):
    # sampling configurations using Langevin dynamics from the strong coupling
    # N_batch = 500
    # U_batch = LG.generate_batch(U_initial=U_batch[0:1,...], eps=eps, N=N_batch, seed=seed).requires_grad_(True)
    # training loop of the wavefunction
    N_epochs = 100
    for e in tqdm(range(N_epochs), leave=False, desc="Training", disable=True):
        optimizer.zero_grad()
        y_SC = psi_SC(U_batch)
        y    = psi(U_batch)
        loss_MSE = ((y-y_SC).abs()**2).mean()
        y_norm = (y.abs()**2).sum()
        y_SC_norm = (y_SC.abs()**2).sum()
        loss_norm = (y_norm - 1)**2
        loss = loss_MSE + loss_norm
        loss.backward()
        optimizer.step()
        print(bs, "|", y_SC_norm.item(), "|", loss_MSE.item(), loss_norm.item(), loss.item())
    #---

