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
#from lattice_data_tools.links.canonical_momenta_squared import La2_Generator

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

n_links = U.n_links

LCNN_layer = LCNN(U=U, K=K)

W = LCNN_layer.get_W(U=U)

N_in = W.shape[-3]
N_out = 2

N_hidden = 2
N_neurons = [4,4]

N_epochs = 500

model = LCNN_MLP(
    U = U,
    LCNN_layer= LCNN_layer,
    LCNN_N_in=N_in, LCNN_N_out = N_out,
    N_hidden = N_hidden, N_neurons = N_neurons,
    seed = seed,
    act_fun_MLP = torch.nn.Tanh()
    )



f_is_real = True
if f_is_real:
    f = lambda U: model(U).real
else:
    f = lambda U: model(U)

print("f.shape", f(U).shape)

# LG = La2_Generator(f=f, U=U, do_compile=True)
# La2_compiled = perf(lambda: LG.df_function(U.as_subclass(torch.Tensor)), "La2 compiled)")

LD = LieDerivatives(U=U)

a_generator = Ng//2

CM2_ad = La2_with_ad(U=U)
CM2_fd = La2_with_fd(U=U)
La2_ad = perf(lambda: CM2_ad.get_La2_per_link(f=f, U=U), "La2_ad")
La2_fd = perf(lambda: CM2_fd.get_La2_per_link(f=f, U=U), "La2_fd")
#LaLa = perf(lambda: CM2_ad.with_La_twice(f=f, U=U), "LaLa")
#La2_fd_fast = perf(lambda: CM2_fd.get_La2_per_link_fast(f=f, U=U), "La2_fd_fast")

f_cplx = lambda U: f(U) + 0.0*1j
CM= CanonicalMomenta(U=GaugeConfiguration(U))
#LaLa_cr = perf(lambda: CM.LaLa_chain_rule_EXPERIMENTAL(f=f_cplx, U=U), "La twice")

#print(LaLa_cr.sum(dim=(1,2,3,4)).real)
#print(LaLa_cr.mean(dim=(1,2,3,4)).real)
#print("Ratio:", LaLa_cr.mean(dim=(1,2,3,4)).real/La2_fd)
print("---")
print(La2_fd)
print("---")
print(La2_ad)
print(torch.allclose(La2_ad, La2_fd))
#print(torch.allclose(La2_ad, LaLa_cr.real))
#print(Ng, LaLa.shape, LaLa.sum(dim=-1)/n_links, La2_ad)

#print(torch.allclose(LaLa, La2_ad[:,0,...]))
# print(La2_ad.flatten()[0:10])
#print(LaLa.flatten()[0:10])
#print(torch.allclose(La2_ad, La2_fd_fast))

t3 = time.time()
La_squared_per_link = perf(lambda: LD.La_squared_per_link(f=f, U=U, f_is_real=f_is_real), f"AD: sum La_squared for all a")
La_squared_per_link_FD = perf(lambda: LD.La_squared_per_link_FD(f=f, U=U, f_is_real=f_is_real),  f"FD: sum La_squared for a={a_generator}")
La_squared_per_link_FD_fast = perf(lambda: LD.La_squared_per_link_FD_fast(a=a_generator, f=f, U=U, f_is_real=f_is_real).to(dtype=La_squared_per_link.dtype), f"FD (fast): sum La_squared for a={a_generator}")


print(La_squared_per_link)
print(La_squared_per_link_FD)
print(torch.allclose(La_squared_per_link, La_squared_per_link_FD))
print(torch.allclose(La_squared_per_link, La_squared_per_link_FD_fast))
print(torch.allclose(La_squared_per_link_FD, La_squared_per_link_FD_fast))


