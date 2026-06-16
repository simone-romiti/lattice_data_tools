"""
IN PROGRESS

Machine Learning a strong coupling eigenstate using Langevin dynamics to check the normalization
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

from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

device = torch.device("cpu")
B = 1
d = 2
L = 3
L_mu = d*[L]
Nc = 2
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260615

U0 = GaugeConfiguration.from_hotstart(
    batchsize=1, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

# U0 = GaugeConfiguration.from_coldstart(
#     batchsize=1, L_mu=L_mu, Nc=Nc,
#     dtype=torch.complex128, device=device,
#     requires_grad=True)

# strong coupling eigenfunction
#psi_SC = lambda Ui: Ui.plaquettes().as_subclass(torch.Tensor).squeeze()/np.sqrt(Nc)
psi_SC = lambda Ui: suN.get_ReTr(Ui.plaquette(torch.tensor([0,0]), mu=0,nu=1))
abs_psi_SC = lambda Ui: psi_SC(Ui).abs()
abs_psi2_SC = lambda Ui: abs_psi_SC(Ui)**2
log_abs_psi2_SC = lambda Ui: 0.0*torch.log(abs_psi2_SC(Ui))+0.0*1j
#log_abs_psi2_SC = lambda Ui: -Ui.Wilson_action(beta=3.0)+0.0*1j

print("Thermalization")
N_therm = 5000
LG = LangevinDynamics(U=U0, log_p=log_abs_psi2_SC)
eps = 1e-2
thermalization = LG.evolve(
    U=U0, eps=eps, N=N_therm, seed=seed,
    omeas = lambda i, Ui: suN.get_ReTr(Ui.plaquette(torch.tensor([0,0]), mu=0,nu=1))
)



Oi = abs_psi2_SC(U0)
Oi = torch.stack(thermalization["Oi"]).view(-1)[-2000:].abs()**2
print(Oi.shape, Oi.mean(), Oi.std())

#plt.title(f"Average plaquette={Oi.mean()} +/- {Oi.std()}")
plt.plot(Oi, label="real")
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

