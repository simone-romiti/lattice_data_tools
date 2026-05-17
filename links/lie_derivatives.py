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
from torch.func import jvp

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration, ColorMatrix



class LieDerivatives:
    """
    Class for the calculation of the Lie derivatives (canonical momenta)
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
        self.omega_glob = torch.zeros(size=(1,), dtype=U.dtype, device=U.device, requires_grad=True)
        omega_shape = (*U.shape[0:-2],) #, self.Ng) # (batch, x,\\mu) indices
        self.omega = torch.zeros(size=omega_shape, dtype=U.dtype, device=U.device, requires_grad=True)
        self.tau = suN.get_generators(Nc=self.Nc, device=U.device, dtype=U.dtype)
        self.V = [suN.get_exp_iA(A = - torch.einsum("...,ij->...ij", self.omega, self.tau[a,:,:])) for a in range(self.Ng)] # exp(-i*omega*tau_a)
    #---

    def L_a(self, a: int, f: typing.Callable, U: GaugeConfiguration):
        d, M = torch.linalg.eigh(self.tau[a,:,:].expand(*self.omega.shape, self.Nc, self.Nc))
        phase = self.omega.unsqueeze(-1) * d
        exp_iD = torch.diag_embed(torch.exp(1j*phase)) #.type(M.type())  # exp(d_k) for each eigenvalue d_k
        Va = M @ exp_iD @ M.adjoint()   # U = M exp(iD) M^\\dagger
        Va_U = GaugeConfiguration(Va @ U)
        f_VaU = f(Va_U).view(-1)
        df_domega_list = []
        batchsize = U.batch_size
        for i in range(batchsize):
            df_domega_Re = torch.autograd.grad(
                    f_VaU[i,0].real,
                    self.omega[i,...],
                    create_graph=True
                )[0]
            df_domega_Im = torch.autograd.grad(
                    f_VaU[i,0].imag,
                    self.omega[i,...],
                    create_graph=True
                )[0]
            df_domega_list.append(df_domega_Re + 1j*df_domega_Im)
        #---
        df_domega = torch.stack(df_domega_list, dim=0)
        print("shape", df_domega.shape)
        return df_domega
        
    def R_a(self, f: typing.Callable[[GaugeConfiguration], ColorMatrix], U: GaugeConfiguration):
        pass
        # f_prime_values = lambda V:  f(U @ V)
        # df_domega = componentwise_autodiff(y=f_prime_values, x=self.omega)
        # return +1j*df_domega



