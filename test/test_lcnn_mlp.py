"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import torch
torch.autograd.set_detect_anomaly(True, check_nan=False)

import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta

from lattice_data_tools.links.canonical_momenta_squared import WithAutodifferentiation as La2_with_ad
from lattice_data_tools.links.canonical_momenta_squared import WithFiniteDifferences as La2_with_fd

from lattice_data_tools.links.lie_derivatives import LieDerivatives
from lattice_data_tools.links.loops import WilsonLoopsGenerator
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
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
K = 0 # L//2
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260511

torch.manual_seed(seed=seed)

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

LCNN_layer = LCNN(U=U, K=K)

W = LCNN_layer.get_W(U=U)

N_in = W.shape[-3]
N_out = 5

N_hidden = 2
N_neurons = [5,5]

N_epochs = 500

model = LCNN_MLP(
    U = U,
    LCNN_layer= LCNN_layer,
    LCNN_N_out = N_out,
    N_hidden = N_hidden, N_neurons = N_neurons,
    seed = seed,
    act_fun_MLP = torch.nn.Tanh()
    )


model.train() # training mode
for i in range(N_epochs):
    print(f"Epoch: {i}/{N_epochs}")
    psi = model(U)

