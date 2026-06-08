"""
The compilation with torch.compile requires that I use all real numbers to be effective.
This file contains a minimal implementation of the canonical momenta,
for a function f(U) that is non-trivial in gauge configuration of links.
The latter is casted into real by extending the last dimension to contain real and imaginary part

"""

import sys

sys.path.append("../../../")

import time
import torch
import typing
import warnings
#warnings.filterwarnings("always")

from lattice_data_tools.links.configuration import GaugeConfiguration
import lattice_data_tools.links.suN as suN
from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function, GradientGenerator
from lattice_data_tools.autodifferentiation.with_torch_func_grad import LaplacianGenerator_new
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta, La_Generator
from lattice_data_tools.links.canonical_momenta_squared import WithAutodifferentiation as La2_with_ad
from lattice_data_tools.links.canonical_momenta_squared import WithFiniteDifferences as La2_with_fd



def complex2ri(x):
    return torch.stack([x.real, x.imag], dim=-1)

def adjoint(M):
    M_T = torch.transpose(M, -3, -2)
    M_H = torch.stack([M_T[..., 0], -M_T[..., 1]], dim=-1)
    return M_H

def conjugate(M):
    return torch.stack([M[..., 0], -M[..., 1]], dim=-1)


def complex_matmul(A, B):
    ReA, ImA = A[..., 0], A[..., 1]
    ReB, ImB = B[..., 0], B[..., 1]
    Re_AB = ReA @ ReB - ImA @ ImB
    Im_AB = ReA @ ImB + ImA @ ReB
    AB = torch.stack([Re_AB, Im_AB], dim=-1)
    return AB



class La_Generator_exp:
    """
    Object generating the momenta L_a, for each a and each link in the configuration, and each configuration

    The output is a lambda function (potentially compiled with `torch.compile`), which takes U in the usual shape (it is flattened internally)
    """
    def __init__(self, a: int, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
        """
        Initialize the lambda function for the La,
        using the function f,
        an example gauge configuration (for the number of links). U.shape==(batchsize, n_links)
        and when `do_compile==True` it is compiled
        """
        batchsize = U.shape[0]
        n_links = U.n_links
        Nc = U.Nc
        Ng = U.Ng
        device = U.device
        dtype = U.dtype

        def g_a(a, eps, U_flat):
            # diagonalization of the tau_a
            tau = suN.get_generators(Nc=Nc, device=device, dtype=dtype)[a]
            tau_eigh_complex = torch.linalg.eigh(tau)
            # keep tau_eigh as a tuple of real tensors -- passed explicitly, not captured
            tau_eigh = (tau_eigh_complex[0], complex2ri(tau_eigh_complex[1]))
            
            d, M = tau_eigh
            eps_d = torch.einsum("il,e->ile", eps, d)
 
            exp_iD = torch.stack(
                [
                    torch.diag_embed(torch.cos(eps_d)),
                    -torch.diag_embed(torch.sin(eps_d))
                ],
                dim=-1)
            Va = complex_matmul(complex_matmul(M, exp_iD), adjoint(M))
            VaU = complex_matmul(Va, U_flat)
            res = f(VaU) 
            return res[0, :]

        def g(eps, U_flat):
            eps_arr = eps.reshape(batchsize, Ng, -1)
            ga_values = torch.vmap(lambda a: g_a(a, eps_arr[:,a,...], U_flat))(torch.arange(Ng))
            return ga_values.sum(dim=0)
        
        def Re_g(eps, U_flat):
            return g(eps, U_flat)[0]
    
        def Im_g(eps, U_flat):
            return g(eps, U_flat)[1]
        

        U_flat = complex2ri(torch.Tensor(U)).view(batchsize, -1, Nc, Nc, 2)
        eps_shape = (batchsize, Ng*n_links) #U_flat.shape[-1])
        eps = torch.zeros(*eps_shape, device=U.device, dtype=U.real.dtype)
        lapl_Re_g = LaplacianGenerator_new(Re_g, eps, U_flat, do_compile=do_compile)
        lapl_Im_g = LaplacianGenerator_new(Im_g, eps, U_flat, do_compile=do_compile)

        def _call(U_conf):
            U_ri = complex2ri(torch.Tensor(U_conf))
            U_flat = U_ri.view(batchsize, -1, Nc, Nc, 2)
            # eps = torch.zeros(*U_flat.shape[:-3], device=U.device, dtype=U.real.dtype)
            return lapl_Re_g.d2f_function(eps,U_flat) + 1j*lapl_Im_g.d2f_function(eps, U_flat)

        self.df_function = lambda U: _call(U_conf=U).reshape(batchsize, Ng, n_links)



def perf(fun, info: str):
    torch.cuda.synchronize()
    t1 = time.time()
    res = fun()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res


def f(U_ri):
    U_xp1 = torch.roll(U_ri, shifts=(1,3), dims=(0,2))
    n = len(U_ri.shape)
    res = (U_xp1 * U_ri).sum(dim=tuple(torch.arange(1, n - 1)))
    return res

def f_from_conf(U):
    ri_res = f(complex2ri(U.as_subclass(torch.Tensor)))
    cmplx_res = (ri_res[:,0] + 1j*ri_res[:,1]).unsqueeze(-1)
    return cmplx_res


f_is_real = False

B = 1
L_mu = [4,2]
Nc = 3
device = torch.device("cpu")

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=20260501, dtype=torch.complex128, device=device,
    requires_grad=False
)

U_tens = U.as_subclass(torch.Tensor)
U_ri = complex2ri(U_tens)

a_ref = 0

LaG = La_Generator_exp(a=a_ref, f=f, U=U, do_compile=False)
La_arr_vmap = perf(lambda: LaG.df_function(U=U), "La not compiled")

LaG_compiled = La_Generator_exp(a=a_ref, f=f, U=U, do_compile=True)
La_arr_vmap_compiled = perf(lambda: LaG_compiled.df_function(U=U), "La2 compiled")

print("La2 from here:", La_arr_vmap.shape, La_arr_vmap_compiled.shape)


CM2_ad = La2_with_ad(U=U)
La2_ad = perf(lambda: CM2_ad.get_La2_per_link(f=f_from_conf, U=U), "La2_ad")
CM2_fd = La2_with_fd(U=U)
La2_fd = perf(lambda: CM2_fd.get_La2_per_link(f=f_from_conf, U=U), "La2_fd")
print("La2 reference: ", La2_ad.shape, La2_fd.shape)

## compare with known implementation
CM = CanonicalMomenta(U=U)
# momenta_exp = perf(lambda: CM.with_exponential(f=f, U=U, f_is_real=f_is_real), "L_a & R_a arr from exp()")
momenta_cr = perf(lambda: CM.with_chain_rule(f=f_from_conf, U=U, f_is_real=f_is_real), "L_a & R_a arr from chain rule")

# print(momenta_exp.shape)
print(momenta_cr.shape)
print(La_arr_vmap.shape, La_arr_vmap_compiled.shape)

#print("L_a (vmap) VS chain rule: ", torch.allclose(momenta_cr[:,0,...], La_arr_vmap))
#print(torch.allclose(La_arr_vmap, La_arr_vmap_compiled))
