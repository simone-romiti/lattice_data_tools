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
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix

def get_exp_i_omega_tau_a(omega, tau_a):
    """
    Returns `V_a = exp(i*omega*tau_a)` through diagonalization.
    This is done by diagonalizing `tau_a` only, as `omega` is a scalar.
    
    NOTE: This version is safer for autodifferentiation at `omega==0`.
    """
    d, M = torch.linalg.eigh(tau_a) # diagonalization of the generator tau_a
    phase = omega * d # angle phases
    exp_iphase = torch.exp(1j*phase) # diagonal matrix from diagonal entries
    exp_iD = torch.diag_embed(exp_iphase) 
    Va = M @ exp_iD @ M.adjoint()   # V_a = M exp(iD) M^\\dagger
    return Va
#---

class LieDerivatives:
    """
    Class for the calculation of the Lie derivatives (canonical momenta of gauge links)
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
            U: GaugeConfiguration object, used to infer the shape of `omega`, `dtype` and `device`.

        where `Ng` is the number of generators of `SU(Nc)`.
        """
        self.Nc = U.Nc # number of colors of the group
        self.Ng = U.Ng # number of generators in the algebra
        # self.omega_glob = torch.zeros(size=(1,), dtype=U.real.dtype, device=U.device, requires_grad=True)
        self.omega_shape = (*U.shape[0:-2],) #, self.Ng) # (batch, x,\\mu) indices
        self.omega = torch.zeros(size=self.omega_shape, dtype=U.real.dtype, device=U.device, requires_grad=True)
        self.tau = suN.get_generators(Nc=self.Nc, device=U.device, dtype=U.dtype)
        # list of exponentials exp(i*omega*tau_a)
        self.V = [
            get_exp_i_omega_tau_a(
                omega=self.omega.unsqueeze(-1),
                tau_a=self.tau[a,:,:]
            ) for a in range(self.Ng)
        ] # exp(-i*omega*tau_a)
    #---

    def L_a(self, a: int, f: typing.Callable, U: GaugeConfiguration):
        tau_a = self.tau[a]  # (Nc, Nc)
        batchsize = U.batch_size
        delta = torch.zeros(size=U.shape[1:], dtype=U.real.dtype, device=U.device, requires_grad=True)
        f_U = f(GaugeConfiguration(U+delta.unsqueeze(0)))
        df_dU_batchlist = []
        for i in range(batchsize):
            Re_df_dU_i = torch.autograd.grad(
                outputs=f_U[i,...].real,
                inputs=delta,
                create_graph=True,
                grad_outputs = torch.ones_like(f_U[i,...].real),
            )[0]
            Im_df_dU_i = torch.autograd.grad(
                outputs=f_U[i,...].imag,
                inputs=delta,
                create_graph=True,
                grad_outputs = torch.ones_like(f_U[i,...].imag),
            )[0]
            df_dU_batchlist.append(Re_df_dU_i + 1j*Im_df_dU_i)
        #---
        df_dU = torch.stack(df_dU_batchlist, dim=0)
        tau_aU = torch.tensor(torch.einsum("ij,...jk->ik", tau_a, U))
        La_fU = - tau_aU * df_dU
        return La_fU
        
    def R_a(self, f: typing.Callable[[GaugeConfiguration], ColorMatrix], U: GaugeConfiguration):
        pass
        # f_prime_values = lambda V:  f(U @ V)
        # df_domega = componentwise_autodiff(y=f_prime_values, x=self.omega)
        # return +1j*df_domega



