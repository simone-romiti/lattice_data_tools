"""
Implementation of Lie derivatives on functions of the gauge links,
using autodifferentiation:

$${ L_a f(U) = -i\\frac{d}{d \\omega} f( e^{-i \\omega \\tau_a } U ) |_{\\omega = 0} }$$
$${ R_a f(U) = +i\\frac{d}{d \\omega} f( U e^{-i \\omega \\tau_a } ) |_{\\omega = 0} }$$

The $L_a$ and $R_a$ are the canonical momenta associated to the link $U$,
and satisfy:

$${ [L_a , U] = - \\tau_a U }$$
$${ [R_a , U] = + U \\tau_a }$$

"""

import typing

import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration

from lattice_data_tools.autodifferentiation.with_torch_func_grad import get_compiled_function


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
            return res[0,0]
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
            

class CanonicalMomenta:
    """
    Class for the calculation of the canonical momenta (i.e. Lie derivatives) in SU(N)
    """
    def __init__(self, U: GaugeConfiguration):
        """
        Initialization of tensors needed
        
        Main idea: this object is initialized before the training,
        and allows to compute efficiently the derivatives with respect to $\\omega_a(x,\\mu)$ in $0$.
        For instance:

        $${ L_a(x,\\mu) f(..., U(x,\\mu), ...) = \\frac{d}{d\\omega} f(..., V_a U(x,\\mu), ...)|_{\\omega=0} }$$

        where ${ V_a = exp(-i \\omega_a(x,\\mu) \\tau_a) }$

        Input:
            U: GaugeConfiguration object, used only to infer the shape of `omega`, `dtype` and `device`.

        where `Ng` is the number of generators of `SU(Nc)`.
        """
        self.Nc = U.Nc # number of colors of the group
        self.Ng = U.Ng # number of generators in the algebra
        self.omega_shape = (*U.shape[1:-2],) # (L1,...,Ld, d): one omega per link, no batch, no color
        self.omega = torch.zeros(size=self.omega_shape, dtype=U.real.dtype, device=U.device, requires_grad=True)
        self.tau = suN.get_generators(Nc=self.Nc, device=U.device, dtype=U.dtype)
        self.V = self.get_V(omega=self.omega, tau=self.tau) # exp(-i*omega*tau_a)
        
    #---
    def get_V(self, omega: torch.Tensor, tau: torch.tensor):
        """
        Returns the tensor `V` made of

        `V = exp(-i*omega(tau[a]))`

        with shape `(Ng, Nc, Nc)`: `Ng` matrices in `SU(Nc)`,
        where `Ng` is the number of generators in the algebra of the group.

        `omega`: tensor with shape (L1,...,Ld,d)
        `tau`: tensor with shape (Ng, Nc, Nc)
        """
        V = suN.get_exp_i_omega_tau_a(
            omega = - omega,
            tau_a =   tau
        )
        return V
    
    def with_exponential(
            self,
            f: typing.Callable, U: GaugeConfiguration, f_is_real: bool):
        """
        (slower implementation compared to the one using the chain rule)

        Left canonical momenta from the definition of the Lie derivatives,
        using autodifferentiation on `omega`.

        $${ L_a f(U) = -i (d/d omega) f(e^{-i* omega *tau_a} @ U) |_{\\omega=0} }$$
        $${ R_a f(U) = +i (d/d omega) f(U @ e^{-i* omega *tau_a}) |_{\\omega=0} }$$

        NOTE:
            For the R_a, we mathematically this is equivalent to the formula with `-i` in front and `+i` in the exponent: at `omega=0` the derivative is the same as for `-omega`.
            Using `exp(-i*omega*tau_a)` for both the `L_a` and `R_a` saves computing time.


        `omega`: tensor with shape (L1,...,Ld,d), value `0` and `requires_grad==True`
        `V`: list of `exp(-i*omega*tau_a)` for all `a`, computed **on the same** `omega` given as this function's input. It can be obtained with `get_V()`
        `f`: scalar function returning tensor of shape (batchsize, 1)
        `U`: batch of gauge configuations
        `f_is_real`: True when the output if `f` is a real number, False when complex.

        """
        Ng = U.Ng # number of generators in the Lie algebra
        B = U.batch_size # number of configurations
        U_prime = torch.stack(
            [
                torch.einsum("...aij,B...jk->Ba...ik", self.V, U),
                torch.einsum("B...ij,...ajk->Ba...ik", U, self.V)
            ],
            dim=1
        ).view(*( (B*2*Ng,) + U.shape[1:] )) # flattening non-geometric dimensions
        f_U = f(GaugeConfiguration(U_prime)) # shape: (b,Ng,1)
        f_U_flat = f_U.view(-1)
        N_tot = f_U_flat.numel()
        if f_is_real:
            df_domega = torch.stack(
                [                
                    torch.autograd.grad(
                        outputs=f_U_flat[i],
                        inputs=self.omega,
                        create_graph=True, # no need to create graph
                        grad_outputs = torch.ones_like(f_U_flat[i])
                    )[0]
                    for i in range(N_tot)
                ],
                dim = 0
            ).reshape(B,2,Ng, *U.shape[1:-2])
        else:
            Re_df_domega = torch.stack(
                [
                    torch.autograd.grad(
                        outputs=f_U_flat[i].real,
                        inputs=self.omega,
                        create_graph=True, # graph needed for imaginary part
                        grad_outputs = torch.ones_like(f_U_flat[i].real),
                    )[0]
                    for i in range(N_tot)
                ],
                dim = 0
            )
            Im_df_domega = torch.stack(
                [
                    torch.autograd.grad(
                        outputs=f_U_flat[i].imag,
                        inputs=self.omega,
                        create_graph=True,
                        grad_outputs = torch.ones_like(f_U_flat[i].imag),
                    )[0]
                    for i in range(N_tot)
                ],
                dim=0
            )
            df_domega = (Re_df_domega + 1j*Im_df_domega).reshape(B,2,Ng, *U.shape[1:-2])
        #---
        imag_unit_factor = torch.tensor([-1j, +1j]) # factor in front: `-i` or `+i`
        momenta = torch.einsum("D,bD...->bD...", imag_unit_factor, df_domega) # shape (B,2,Ng,...), where index 1 is for Left(0) or Right(1) momenta
        return momenta
    
    def with_chain_rule(
            self,
            f: typing.Callable,
            U: GaugeConfiguration,
            f_is_real: bool):
        """
        (faster implementation compared to the one using the exponentials of the generators)

        Action of Left and Right canonical momenta of a scalar function f(U), using the Chain Rule.
        It returns a tensor of shape (batchsize, 2, a, ..., Nc, Nc). "2" is for either Left or Right momentum

        $$
        L_a f(U) =
        -i (d/d omega) f(e^{-i* omega *tau_a} @ U) |_{omega=0} =
        -i * (-i*\\tau_a U)_{ab} (\\partial f / \\partial U_{ab})
        $$

        $$
        R_a f(U) =
        -i (d/d omega) f(U @ e^{+i* omega *tau_a}) |_{omega=0} =
        -i * (+i U \\tau_a U)_{ab} (\\partial f / \\partial U_{ab})
        $$

        Notes:
            1. The function `f` returns a shape (batchsize, 1): one function value for each configuration
            2. `df/dU=df(U+\\delta)/d\\delta |_{\\delta=0}`, with `delta` complex
            3. If `w=w1+i*w2`, `z=z1+iz2` --> `w1*z1+w2+z2=Re(w*conj(z))`    
            4. Here we compute -i d/domega with the chain rule over the real and imaginary parts of the color indices of the arguments

            For `f=f_1+i*f_2`, `i=1,2`, this gives (e.g. for the `L_a`):

            $$
               (df/d_\\omega)|_{\\omega=0} =
               dRe(-1j*\\tau_a*U)_{ab} Re( df/dRe(U_{ab})) + dIm(-1j*\\tau_a*U_{ab}) Im( df/dIm(U_{ab})) =
               Re[ (-1j*\\tau_a*U) * conj(df_i/dU) ]
            $$
        """
        # tau_a = self.tau[a,:,:]  # (Nc, Nc)
        delta = torch.zeros(size=U.shape[1:], dtype=U.dtype, device=U.device, requires_grad=True) # Note: delta should be complex
        f_U = f(GaugeConfiguration(U + delta.unsqueeze(0))) # f(U+delta)
        assert(f_U.shape == (U.batch_size, 1)) # scalar output, one for each batch 
        batchsize = f_U.numel() # f(U) is a scalar --> one element for each batch

        Nc = U.Nc # number of colors
        tau = suN.get_generators(Nc=Nc, device=U.device, dtype=U.dtype) # su(N) algebra generators
        # d(e^{-i*omega*tau_a} U)/domega at omega==0
        # shape: (batchsize, a, ..., Nc, Nc)
        A_L = -1j * torch.einsum("aij,b...jk->b...aik", tau, U.as_subclass(torch.Tensor))
        # d(U e^{+i*omega*tau_a})/domega at omega==0
        # shape: (batchsize, a, ..., Nc, Nc)
        A_R = -1j * torch.einsum("b...ij,ajk->b...aik", U.as_subclass(torch.Tensor), tau)
        A = torch.stack((A_L, A_R), dim=1) # (batchsize, 2, a, ..., Nc, Nc) | "2" is for either Left or Right momentum

        batchsize = U.batch_size
        if f_is_real:
            df_dU = torch.stack(
                [
                    torch.autograd.grad(
                        outputs=f_U[i],
                        inputs=delta,
                        create_graph=False
                    )[0] # graph not needed later                    
                    for i in range(batchsize)
                ],
                dim=0
            )
            # for each batch, sum over color indices
            df = torch.einsum("b...ij,bD...aij->bDa...", df_dU.conj(), A).real # df/domega
        else:
            Re_df_dU = torch.stack(
                [
                    torch.autograd.grad(
                        outputs=f_U[i].real,
                        inputs=delta,
                        create_graph=True,
                        retain_graph=True,
                    )[0] # real part of df(U)/dU
                    for i in range(batchsize)
                ],
                dim=0
            )
            Im_df_dU = torch.stack(
                [
                    torch.autograd.grad(
                        outputs=f_U[i].imag,
                        inputs=delta,
                        create_graph=True,
                        retain_graph=True
                    )[0] # imaginary part of df(U)/dU
                    for i in range(batchsize)
                ]
            )
            df_re = torch.einsum("b...ij,bD...aij->bDa...", Re_df_dU.conj(), A).real # d Re(f)/domega
            df_im = torch.einsum("b...ij,bD...aij->bDa...", Im_df_dU.conj() , A).real # d Im(f)/domega
            df_domega = df_re + 1j * df_im # d( Re(f) + i*Im(f) )/domega
        #---
        imag_unit_factor = torch.tensor([-1j, +1j]) # factor in front: `-i` or `+i`
        momenta = torch.einsum("D,bD...->bD...", imag_unit_factor, df_domega) # shape (B,2,Ng,...), where index 1 is for Left(0) or Right(1) momenta
        return momenta
