"""
SU(N) canonical momenta using `torch.func.grad`.
These routines served for:

- testing against the `torch.autograd.grad`
- proving an measuring the performance improvement gained using `torch.compile`

NOTE: In my tests, the compilation led to an improvement, but still below the performance of an uncompiled `torch.autograd.grad`.

"""

import torch
import typing

from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function
from lattice_data_tools.links.configuration import GaugeConfiguration
import lattice_data_tools.links.suN as suN


class La_Generator:
    """
    Object generating the momenta L_a^2, for each a and each link in the configuration, and each configuration

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
        batchsize = U.batch_size # number of configurations
        n_links = U.n_links # number of links
        Nc = U.Nc # number of colors
        Ng = U.Ng # number of generators of the Lie algebra
        lattice_shape = U.lattice_shape
        device = U.device
        dtype = U.dtype

        # diagonalization of the tau_a
        tau = suN.get_generators(Nc=Nc, device=device, dtype=dtype)
        tau_eigh = torch.linalg.eigh(tau)

        Id = torch.eye(Nc).to(device=U.device)
        Id_arr = Id.expand(n_links, Nc, Nc)
        single_conf_shape = (1,)+U.shape[1:]
        def f_shift(tau_a_eigh,  Ub, eps, i):
            # d, M = tau_eigh[a]
            d, M = tau_a_eigh # torch.linalg.eigh(tau_a)
            exp_iD = torch.diag_embed(torch.exp(-1j * eps * d))
            Va = M @ exp_iD @ M.adjoint()
            ei = (torch.arange(n_links, device=Ub.device) == i).to(Ub.dtype)
            Va_arr = Id_arr + torch.einsum("ab,i->iab", Va-Id, ei)
            VaU_i = (Va_arr @ Ub).reshape(*single_conf_shape)
            res = f(GaugeConfiguration(VaU_i)) # shape==(1,1)
            return res[0,0] + 0.0*1j
        def Re_f_shift(tau_a_eigh, Ub, eps, i):
            return f_shift(tau_a_eigh, Ub, eps, i).real
        def Im_f_shift(tau_a_eigh, Ub, eps, i):
            return f_shift(tau_a_eigh, Ub, eps, i).imag

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
            
            def uncompiled_df(U_arr):
                U_flat = U_arr.view(batchsize,-1,Nc,Nc)
                res =  df_i_vmapped(tau_eigh, U_flat, eps, idx_links)
                return res.reshape(La_f_shape)

            U_tens = U.as_subclass(torch.Tensor)
            dummy = uncompiled_df(U_tens) # necessary dummy call to trigger compilation
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

