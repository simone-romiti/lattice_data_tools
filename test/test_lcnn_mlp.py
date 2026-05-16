"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import numpy as np
import torch
import time
import sys
sys.path.append("../../")

# import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
# from lattice_data_tools.links.loops import WilsonLoopsGenerator
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

print("===============================")
print("L-CNN + MLP implementation test")
print("===============================")

device = torch.device("cpu")
B = 1
d = 2
L = 5
L_mu = d*[L]
K = L//2
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260511

#torch.manual_seed(seed=seed)

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=seed, dtype=torch.complex128, device=device, requires_grad=True)


LCNN_layer = LCNN(U=U, K=K)

W = LCNN_layer.get_W()

N_in = W.shape[-3]
N_out = 15

N_hidden = 2
N_neurons = [10,10,10]

N_epochs = 500
model = LCNN_MLP(
    LCNN_layer= LCNN_layer,
    LCNN_N_in=N_in, LCNN_N_out = N_out,
    N_hidden = N_hidden, N_neurons = N_neurons,
    seed = seed,
    act_fun_MLP = torch.nn.Tanh()
    )


model.train() # training mode
for i in range(N_epochs):
    print(f"Epoch: {i}/{N_epochs}")
    psi = model(U)

