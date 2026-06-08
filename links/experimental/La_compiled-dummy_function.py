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
from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function
from lattice_data_tools.links.canonical_momenta import CanonicalMomenta, La_Generator


def complex2ri(x):
    return torch.stack([x.real, x.imag], dim=-1)

def adjoint(M):
    M_T = torch.transpose(M, -3, -2)
    M_H = torch.stack([M_T[..., 0], -M_T[..., 1]], dim=-1)
    return M_H

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
    def __init__(self, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
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

        # diagonalization of the tau_a
        tau = suN.get_generators(Nc=Nc, device=device, dtype=dtype)
        tau_eigh_complex = torch.linalg.eigh(tau)
        # keep tau_eigh as a tuple of real tensors -- passed explicitly, not captured
        tau_eigh = (tau_eigh_complex[0], complex2ri(tau_eigh_complex[1]))

        Id = torch.stack(
            [
                torch.eye(Nc, device=U.device, dtype=U.real.dtype),
                torch.zeros(Nc, Nc, device=U.device, dtype=U.real.dtype)
            ], dim=-1)
        Id_arr = Id.expand(n_links, Nc, Nc, 2)
        single_conf_shape = (1,) + U.shape[1:] + (2,)

        def f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            d, M = tau_a_eigh
            exp_iD = torch.stack(
                [
                    torch.diag_embed(torch.cos(eps * d)),
                    -torch.diag_embed(torch.sin(eps * d))
                ],
                dim=-1)
            Va = complex_matmul(complex_matmul(M, exp_iD), adjoint(M))
            ei = (torch.arange(n_links, device=Ub.device) == i).to(Ub.dtype)
            Va_arr = Id_arr + torch.einsum("abC,i->iabC", Va - Id_arr[0], ei)
            VaU_i = complex_matmul(Va_arr, Ub).reshape(*single_conf_shape)
            res = f(VaU_i)  # shape==(1,1,...,2)
            return res[0, :]

        def Re_f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            return f_shift(tau_a_eigh, Ub, eps, i, Id_arr)[0]

        def Im_f_shift(tau_a_eigh, Ub, eps, i, Id_arr):
            return f_shift(tau_a_eigh, Ub, eps, i, Id_arr)[1]

        # grad wrt eps (argnums=2), Id_arr passed explicitly (argnums=4)
        Re_df = torch.func.grad(Re_f_shift, argnums=2)
        Im_df = torch.func.grad(Im_f_shift, argnums=2)

        # Re_df = torch.func.grad(torch.func.grad(Re_f_shift, argnums=2), argnums=2)
        # Im_df = torch.func.grad(torch.func.grad(Im_f_shift, argnums=2), argnums=2)

        
        eps = torch.tensor(0.0, device=device, dtype=U.real.dtype)
        idx_links = torch.arange(n_links, device=device)
        La_f_shape = (batchsize, Ng, *(U.shape[1:-2]))

        def make_vmapped(df_i):
            return torch.func.vmap(
                torch.func.vmap(
                    torch.func.vmap(
                        df_i,
                        in_dims=(None, None, None, 0, None)  # over link index
                    ),
                    in_dims=(0, None, None, None, None)       # over generators
                ),
                in_dims=(None, 0, None, None, None)           # over batch
            )

        Re_df_vmapped = make_vmapped(Re_df)
        Im_df_vmapped = make_vmapped(Im_df)

        # Single function combining Re and Im -- this is what we compile as one graph
        def uncompiled_df(U_arr, tau_eigh, eps, idx_links, Id_arr):
            U_flat = U_arr.view(batchsize, -1, Nc, Nc, 2)
            Re = Re_df_vmapped(tau_eigh, U_flat, eps, idx_links, Id_arr)
            Im = Im_df_vmapped(tau_eigh, U_flat, eps, idx_links, Id_arr)
            # factor of -i: -i*(Re + i*Im) = Im - i*Re, kept as real tensor (Re, Im parts)
            result_re = Im.reshape(La_f_shape)
            result_im = -Re.reshape(La_f_shape)
            return result_re, result_im

        U_tens = complex2ri(U.as_subclass(torch.Tensor))

        if do_compile:
            compiled_df = get_compiled_function(
                uncompiled_df, U_tens, tau_eigh, eps, idx_links, Id_arr
            )
        else:
            compiled_df = uncompiled_df

        # Store everything needed at call time
        self._compiled_df = compiled_df
        self._tau_eigh = tau_eigh
        self._eps = eps
        self._idx_links = idx_links
        self._Id_arr = Id_arr

    @property
    def df_function(self):
        """
        Returns a callable that takes U_ri (real tensor) and returns
        the canonical momenta as a complex tensor of shape (batchsize, Ng, *lattice_shape).
        """
        def _call(U):
            re, im = self._compiled_df(
                U, self._tau_eigh, self._eps, self._idx_links, self._Id_arr
            )
            return re + 1j * im
        return _call


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
L_mu = [2,2]
Nc = 3
device = torch.device("cpu")

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=20260501, dtype=torch.complex128, device=device,
    requires_grad=False
)

U_tens = U.as_subclass(torch.Tensor)
U_ri = complex2ri(U_tens)


LaG = La_Generator_exp(f=f, U=U, do_compile=False)
La_arr_vmap = perf(lambda: LaG.df_function(U=U_ri), "La not compiled")

LaG_compiled = La_Generator_exp(f=f, U=U, do_compile=True)
La_arr_vmap_compiled = perf(lambda: LaG_compiled.df_function(U=U_ri), "La compiled")

## compare with known implementation

CM = CanonicalMomenta(U=U)
# momenta_exp = perf(lambda: CM.with_exponential(f=f, U=U, f_is_real=f_is_real), "L_a & R_a arr from exp()")
momenta_cr = perf(lambda: CM.with_chain_rule(f=f_from_conf, U=U, f_is_real=f_is_real), "L_a & R_a arr from chain rule")

# print(momenta_exp.shape)
print(momenta_cr.shape)
print(La_arr_vmap.shape, La_arr_vmap_compiled.shape)


print("L_a (vmap) VS chain rule: ", torch.allclose(momenta_cr[:,0,...], La_arr_vmap))
print(torch.allclose(La_arr_vmap, La_arr_vmap_compiled))
