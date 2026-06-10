"""
Testing the Langevin dynamics routines
"""

import os
import matplotlib.pyplot as plt
import torch
torch.autograd.set_detect_anomaly(True, check_nan=False)

import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.langevin import LangevinDynamics

def perf(fun, info: str):
    t1 = time.time()
    res = fun()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res


print("=============================")
print("Testing the Langevin dynamics")
print("=============================")

device = torch.device("cpu")
B = 1
d = 3
L = 4
L_mu = d*[L]
K = 0 # L//2
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260609


print(f"device={device}")
print(f"batchsize={B}")
print(f"(L1,...,Ld)={L_mu}")
print(f"Nc={Nc}, Ng={Ng}")


torch.manual_seed(seed=seed)

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

# U = GaugeConfiguration.from_coldstart(
#     batchsize=B, L_mu=L_mu, Nc=Nc,
#     dtype=torch.complex128, device=device,
#     requires_grad=True)



beta = 5.0

# Langevin dynamics

eps = 0.01
N_evol = 500

psi2 = lambda U: torch.exp(-U.Wilson_action(beta=beta))

log_psi2 = lambda U: torch.log(psi2(U))+0.0*1j
log_psi2 = lambda U: -U.Wilson_action(beta=beta) + 0.0*1j

LG = LangevinDynamics(U=U, log_p=log_psi2)

U_path = "./U_langevin.pt"
if os.path.exists(U_path):
    U = GaugeConfiguration.load(path=U_path)

U_batch = LG.generate_batch(U_initial=U, eps=eps, N=14, seed=seed)
print("Generated batch of configurations:", U_batch.shape)


def omeas(i, Ui):
    res = GaugeConfiguration(Ui).average_ReTr_plaquette().item()
    print(i, res)
    return res

langevin_evolution = LG.evolve(
    U=U, eps=eps, N=N_evol, seed=seed,
    omeas = omeas
)

U_new = langevin_evolution["U"]
U_new.save(path=U_path)

avg_ReTr_plaq_list = langevin_evolution["Oi"]
avg_ReTr_plaq = torch.tensor(avg_ReTr_plaq_list)
print(torch.mean(avg_ReTr_plaq), torch.std(avg_ReTr_plaq))

plt.plot(torch.arange(N_evol).detach(), avg_ReTr_plaq.detach())
plt.savefig("./langevin_evolution.pdf")

