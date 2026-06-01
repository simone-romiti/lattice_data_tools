"""
The compilation woth torch.compile requires that I use all real numbers to be effective.
This file contains a minimal implementation of the canonical momenta,
for a function f(U) that is non-trivial in gauge configuration of links.
The latter is casted into real by extending the last dimension to contain real and imaginary part

"""

import sys
sys.path.append("../../")

import time
import torch
import typing
#import warnings
#warnings.filterwarnings("always")
import torch._dynamo
torch._dynamo.config.verbose = True  # prints every graph break with reason


from lattice_data_tools.links.configuration import GaugeConfiguration
import lattice_data_tools.links.suN as suN
from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function


def complex2ri(x):
    return torch.stack([x.real,x.imag], dim=-1)

def adjoint(M):
    M_T = torch.transpose(M, -3, -2)
    M_H = torch.stack([M_T[...,0], -M_T[...,1]], dim=-1)
    return M_H

def complex_matmul(A, B):
    ReA, ImA = A[..., 0], A[..., 1]
    ReB, ImB = B[..., 0], B[..., 1]
    Re_AB = ReA@ ReB - ImA @ ImB
    Im_AB = ReA@ ImB + ImA @ ReB
    AB = torch.stack([Re_AB, Im_AB], dim=-1)
    return AB


class La_Generator:
    """
    Object generating the momenta L_a, for each a and each link in the configuration, and each configuration

    The output is a lambda function (potentially compiled with `toch.compile`), which takes U in the usual shape (it is flattened iternally)

    """
    def __init__(self, f: typing.Callable, U: GaugeConfiguration, do_compile: bool):
        """
        Initialize the lambda function for the La^2,
        using the function,
        an example gauge configuration (for the number of links). U.shape==(batchsize, n_links)
        and when `do_compile==True` it is compiled
        """
        #assert(len(U.shape) == 2) # (batchsize, n_var)
        batchsize = U.shape[0] # number of configurations
        n_links = U.n_links # number of links
        Nc = U.Nc # number of colors
        Ng = U.Ng # number of generators of the Lie algebra
        lattice_shape = U.lattice_shape
        device = U.device
        dtype = U.dtype

        # diagonalization of the tau_a
        tau = suN.get_generators(Nc=Nc, device=device, dtype=dtype)
        tau_eigh_complex = torch.linalg.eigh(tau)
        tau_eigh = (tau_eigh_complex[0], complex2ri(tau_eigh_complex[1]))

        Id = torch.stack(
            [
                torch.eye(Nc, device=U.device, dtype=U.real.dtype),
                torch.zeros(Nc, Nc, device=U.device, dtype=U.real.dtype)
            ], dim=-1)
        Id_arr = Id.expand(n_links, Nc, Nc, 2)
        single_conf_shape = (1,)+U.shape[1:]+(2,)
        def f_shift(tau_a_eigh,  Ub, eps, i):
            # d, M = tau_eigh[a]
            d, M = tau_a_eigh # torch.linalg.eigh(tau_a)
            exp_iD = torch.stack(
                [
                    torch.diag_embed(torch.cos(eps * d)),
                    torch.diag_embed(torch.sin(eps * d))
                ],
                dim=-1)
            Va = complex_matmul(complex_matmul(M, exp_iD), adjoint(M))
            ei = (torch.arange(n_links, device=Ub.device) == i).to(Ub.dtype)
            Va_arr = Id_arr + torch.einsum("abC,i->iabC", Va-Id, ei)
            VaU_i = complex_matmul(Va_arr, Ub).reshape(*single_conf_shape)
            res = f(VaU_i) # shape==(1,1)
            return res[0,:]
        def Re_f_shift(tau_a_eigh, Ub, eps, i):
            return f_shift(tau_a_eigh, Ub, eps, i)[0]
        def Im_f_shift(tau_a_eigh, Ub, eps, i):
            return f_shift(tau_a_eigh, Ub, eps, i)[1]

        Re_df = torch.func.grad(Re_f_shift, argnums=2) # abstract gradient object
        Im_df = torch.func.grad(Im_f_shift, argnums=2) # abstract gradient object

        eps = torch.tensor(0.0, device=device, dtype=U.real.dtype)
        idx_links = torch.arange(n_links, device=device)
        # idx_generators = torch.arange(Ng, device=device)
        La_f_shape = (batchsize,Ng,*(U.shape[1:-2]))

        def get_compiled(df_i):
            # vmap can parallelize only along dimension with the same size
            df_i_vmapped = torch.func.vmap(
                torch.func.vmap(
                    torch.func.vmap(
                        df_i,
                        in_dims=(None,None,None, 0) # parallelizing only over the variable index --> \\partial_{eps_i}^2 f(V_eps @ U)
                    ),
                    in_dims=(0,None,None,None) # parallelize along generators a=1,...,Ng
                ),
                in_dims=(None,0,None,None),  # parallelizing only over the batch index f(x1^{i}, x2^{i},...)
            )
            # df_i_vmapped = torch.func.vmap(          # over batch
            #     torch.func.vmap(                     # over generators
            #         torch.func.vmap(df_i,            # over link index i
            #                         in_dims=(None, None, None, 0)
            #                         ),
            #         in_dims=(0, None, None, None)
            #     ),
            #     in_dims=(None, 0, None, None)
            # )
            
            def uncompiled_df(U_arr):
                U_flat = U_arr.view(batchsize,-1,Nc,Nc,2)
                res =  df_i_vmapped(tau_eigh, U_flat, eps, idx_links)
                return res.reshape(La_f_shape)

            U_tens = complex2ri(U.as_subclass(torch.Tensor))
            dummy = uncompiled_df(U_tens)
            if do_compile:
                compiled_df_i = get_compiled_function(uncompiled_df, U_tens)
            else:
                compiled_df_i = uncompiled_df
            #---
            return compiled_df_i

        compiled_Re_df = get_compiled(Re_df)
        compiled_Im_df = get_compiled(Im_df)
        # including the factor "i": -i*d/d\\epsilon
        self._df_function = lambda U: -1j*(compiled_Re_df(U) + 1j*compiled_Im_df(U))

    @property
    def df_function(self):
        return self._df_function




def perf(fun, info: str):
    torch.cuda.synchronize()
    t1 = time.time()
    res = fun()
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"dt ({info}): {t2-t1} sec.")
    return res




def f(U_ri):
    n = len(U_ri.shape)
    # U_xp1 = torch.roll(U_ri, shifts=1, dims=0)
    # res = (U_xp1*U_ri).sum(dim=tuple(torch.arange(1,n-1)))
    res = (U_ri).sum(dim=tuple(torch.arange(1,n-1)))
    return res

B = 5
L_mu = [2,2]
Nc = 3
#device = torch.device("cpu")
device = torch.device("cpu")

U = GaugeConfiguration.from_hotstart(
    batchsize=B, L_mu=L_mu, Nc=Nc,
    seed=20260501, dtype=torch.complex128, device=device,
    requires_grad=False
)

U_tens = U.as_subclass(torch.Tensor)
U_ri = complex2ri(U_tens)


LaG = La_Generator(f=f, U=U, do_compile=False)

N = 10

t0 = time.time()
for i in range(N):
    _ = LaG.df_function(U=U_ri)
t1 = time.time()
print((t1-t0)/N)

LaG_compiled = La_Generator(f=f, U=U, do_compile=True)
t0 = time.time()
for i in range(N):
    _ = LaG_compiled.df_function(U=U_ri)
t1 = time.time()
print((t1-t0)/N)


#La_arr_vmap_compiled = perf(lambda: LaG_compiled.df_function(U=U_ri), "La compiled")
# #print(La_arr_vmap)


