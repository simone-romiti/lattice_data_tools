"""
Testing the implementation of the Lattice Convolutional Neural Network (L-CNN)

https://arxiv.org/pdf/2012.12901

"""

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True, check_nan=False)

import time
import sys
sys.path.append("../../")

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration
from lattice_data_tools.links.lie_derivatives import LieDerivatives
from lattice_data_tools.links.loops import WilsonLoopsGenerator
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

print("===============================")
print("L-CNN + MLP implementation test")
print("===============================")

device = torch.device("cpu")
B = 7
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
    seed=seed, dtype=torch.complex128, device=device,
    requires_grad=True)

LCNN_layer = LCNN(U=U, K=K)

W = LCNN_layer.get_W(U=U)

N_in = W.shape[-3]
N_out = 5

N_hidden = 2
N_neurons = [10,10]

N_epochs = 500

model = LCNN_MLP(
    U = U,
    LCNN_layer= LCNN_layer,
    LCNN_N_in=N_in, LCNN_N_out = N_out,
    N_hidden = N_hidden, N_neurons = N_neurons,
    seed = seed,
    act_fun_MLP = torch.nn.Tanh()
    )



LD = LieDerivatives(U=U)

print("First derivative")

# N_test = 50
# for c in range(Ng):
#     tau_a = suN.get_generators(Nc=Nc, device=U.device, dtype=U.dtype)[c,:,:]
#     tau_aU = torch.Tensor(torch.einsum("ij,...jk->...ik", tau_a, U) )
#     for a in range(Nc):
#         for b in range(Nc):
#             for i in range(N_test):
#                 g = lambda U: torch.Tensor(U[...,a,b]).view(*(U.batch_size,-1))[:,i].unsqueeze(-1)
#                 La_gU_old = LD.L_a_old(a=c, f=g, U=U).view(*(U.batch_size,-1))[:,i].unsqueeze(-1)
#                 La_gU = LD.L_a(a=c, f=g, U=U).view(*(U.batch_size,-1))[:,i].unsqueeze(-1)
#                 LHS = La_gU
#                 RHS = - tau_aU[...,a,b].view(*(U.batch_size,-1))[:,i].unsqueeze(-1)
#                 if not (torch.allclose(LHS, RHS) and torch.allclose(La_gU, La_gU_old)):
#                     print(
#                         c,
#                         torch.allclose(LHS, RHS),
#                         torch.allclose(La_gU, La_gU_old)
#                     )




#La_fU = lambda U: LD.L_a(a=0, f=model, U=U)
## La_fU_value = La_fU(U=U)
# f = model
def f(U_conf: GaugeConfiguration):
    P = WilsonLoopsGenerator.plaquettes(U=U_conf)
    TrP = suN.get_Tr(P)
    return torch.Tensor(torch.sum(TrP, dim=(-3,-2,-1))).unsqueeze(-1)

print(f(U).shape)
La_fU_1 = LD.L_a(a=0, f=f, U=U)
La_fU_2 = LD.L_a_old(a=0, f=f, U=U)

print(La_fU_1.shape, La_fU_2.shape)
# print(La_fU_1)
print(torch.allclose(La_fU_1 , La_fU_2))

# print("Second derivative")
# L2a_fU = LD.L_a(a=0, f=La_fU, U=U)

# model.train() # training mode
# for i in range(N_epochs):
#     print(f"Epoch: {i}/{N_epochs}")
#     psi = model(U)

