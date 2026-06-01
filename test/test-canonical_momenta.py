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
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta, La_Generator

from lattice_data_tools.links.canonical_momenta_squared import WithAutodifferentiation as La2_with_ad
from lattice_data_tools.links.canonical_momenta_squared import WithFiniteDifferences as La2_with_fd

from lattice_data_tools.links.lie_derivatives import LieDerivatives
from lattice_data_tools.links.loops import WilsonLoopsGenerator

from lattice_data_tools.machine_learning.lcnn import LCNN
from lattice_data_tools.machine_learning.lcnn_mlp import LCNN_MLP

def perf(fun, info: str):
    t1 = time.time()
    res = fun()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res


print("===========================")
print("Testing the Lie derivatives")
print("===========================")

device = torch.device("cpu")
B = 1
d = 3
L = 3
L_mu = d*[L]
K = 0 # L//2
Nc = 3
t1 = time.time()
Ng = Nc**2 - 1
seed = 20260511


print(f"device={device}")
print(f"batchsize={B}")
print(f"(L1,...,Ld)={L_mu}")
print(f"Nc={Nc}, Ng={Ng}")


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

f_is_real = False
if f_is_real:
    f = lambda U: model(U).real
else:
    f = lambda U: model(U)

print("f.shape", f(U).shape)


LaG = La_Generator(f=f, U=U, do_compile=False)
La_arr_vmap = perf(lambda: LaG.df_function(U=torch.Tensor(U)), "La compiled")
print(La_arr_vmap.shape)


#a_generator = Ng//2
CM = CanonicalMomenta(U=U)
momenta_exp = perf(lambda: CM.with_exponential(f=f, U=U, f_is_real=f_is_real), "L_a & R_a arr from exp()")
momenta_cr = perf(lambda: CM.with_chain_rule(f=f, U=U, f_is_real=f_is_real), "L_a & R_a arr from chain rule")

print(momenta_exp.shape)
print(momenta_cr.shape)

print("L_a & R_a check: ", torch.allclose(momenta_exp, momenta_cr))
print("L_a: ", torch.allclose(momenta_exp[:,0,...], La_arr_vmap))




