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
from lattice_data_tools.links.lie_derivatives import LieDerivatives
from lattice_data_tools.links.loops import WilsonLoopsGenerator
# from lattice_data_tools.links.parallel_transport import get_ParallelTransporters, get_W_shifted
from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

print("===============================")
print("L-CNN + MLP implementation test")
print("===============================")

device = torch.device("cpu")
B = 1
d = 3
L = 4
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





f_is_real = True
if f_is_real:
    f = lambda U: model(U).real
else:
    f = lambda U: model(U)

print("f.shape", f(U).shape)

a_generator = Ng//2


CM = CanonicalMomenta(U=U)
t1 = time.time()
momenta_cr = CM.with_chain_rule(f=f, U=U, f_is_real=f_is_real)
t2 = time.time()
print(f"t2-t1: {t2-t1} sec.")

t2 = time.time()
momenta_exp = CM.with_exponential(f=f, U=U, f_is_real=f_is_real)
t3 = time.time()
print(f"t3-t2: {t3-t2} sec.")

La_fU_1 = LD.L_a(a=a_generator, f=f, U=U, f_is_real=f_is_real)
La_fU_2 = LD.L_a_chain_rule(a=a_generator, f=f, U=U, f_is_real=f_is_real)
print(momenta_exp.shape)
print("L_a check: ", torch.allclose(La_fU_1 , La_fU_2))
print("L_a check 2: ", torch.allclose(La_fU_1 , momenta_cr[:,0,a_generator,...]))
print("L_a check 2: ", torch.allclose(La_fU_1 , momenta_exp[:,0,a_generator,...]))




t3 = time.time()
La_squared_per_link = LD.La_squared_per_link(a=a_generator, f=f, U=U, f_is_real=f_is_real)
t4 = time.time()
print(f"t4-t3: {t4-t3} sec.")


t4 = time.time()
La_squared_per_link_FD = LD.La_squared_per_link_FD(a=a_generator, f=f, U=U, f_is_real=f_is_real).to(dtype=La_squared_per_link.dtype)
t5 = time.time()
print(f"t5-t4: {t5-t4} sec.")

t5 = time.time()
La_squared_per_link_FD_fast = LD.La_squared_per_link_FD_fast(a=a_generator, f=f, U=U, f_is_real=f_is_real).to(dtype=La_squared_per_link.dtype)
t6 = time.time()
print(f"t6-t5: {t6-t5} sec.")
# t6 = time.time()
# La_squared_per_link_funcgrad = (
#     LD.La_squared_per_link_funcgrad(
#         a=a_generator,
#         f=f,
#         U=U,
#         f_is_real=f_is_real,
#     )
# )
# t7 = time.time()
# print(f"t7-t6: {t7-t6} sec.")

print(torch.allclose(La_squared_per_link, La_squared_per_link_FD))
print(torch.allclose(La_squared_per_link, La_squared_per_link_FD_fast))
print(torch.allclose(La_squared_per_link_FD, La_squared_per_link_FD_fast))


print(La_squared_per_link, La_squared_per_link_FD, La_squared_per_link_FD_fast)
print(torch.allclose(La_squared_per_link, La_squared_per_link_FD_fast))
print(torch.allclose(La_squared_per_link_FD, La_squared_per_link_FD_fast))
# print(torch.allclose(La_squared_per_link, La_squared_per_link_funcgrad)) # 

#print(La_fU_1.shape, La_fU_2.shape, La_squared_per_link.shape)
# print(La_fU_1)
#print(La_squared_per_link.shape)


# print("Second derivative")
# L2a_fU = LD.L_a(a=0, f=La_fU, U=U)

# model.train() # training mode
# for i in range(N_epochs):
#     print(f"Epoch: {i}/{N_epochs}")
#     psi = model(U)

