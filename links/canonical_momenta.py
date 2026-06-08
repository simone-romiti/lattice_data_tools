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
import time
import torch

import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.configuration import GaugeConfiguration

from lattice_data_tools.autodifferentiation.with_torch_autograd_grad import my_autograd


def chain_rule_contributions(A, dRef_dU, dImf_dU):
    """
    Chain rule contributions for the derivative of a function `f(U(omega)) = Re(f) + 1j*Im(f)`:
    `df_domega = (dU/domega) * (df/dU)`
    where d/dU is the Wirtiger derivative: `d/dU = d/dRe(U) - 1j * d/dIm(U)`

    NOTE: torch.autograd.grad returns d/dU^{*}, so you need to pass the `.conj()` of what you get with that function.
    Example:
      dRef_dUstar = torch.autograd.grad(...).conj()
      dImf_dUstar = torch.autograd.grad(...).conj()
      df_domega = autograd_chain_rule(A, df_dUstar)

    By interpreting f(U) as f(Re(U), Im(U)), if A=dU/domega, the chain rule gives:

    `df_domega = \\sum_i [Re(A)]_i * df/dRe(U)_i + [Im(A)]_i * df/dIm(U)_i `

    This function returns an array with the i-th terms of the sum.

    Since f = f_1  + 1j*f_2, we have:

    - `df_domega = df_1/domega + 1j*df_2/domega`
    - `df_i/d_omega = Re(A * df_i/dU)`.

    """
    df_re = (dRef_dU * A).real # d Re(f)/domega
    df_im = (dImf_dU * A).real # d Im(f)/domega
    return df_re + 1j * df_im # d( Re(f) + i*Im(f) )/domega


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
        self.batchsize = U.batch_size #  number of configurations
        self.tau = suN.get_generators(Nc=self.Nc, device=U.device, dtype=U.dtype)
        
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
    
    def LaRa_with_exp(
            self,
            f: typing.Callable, U: GaugeConfiguration, f_is_real: bool):
        """

        Left canonical momenta from the definition of the Lie derivatives,
        using autodifferentiation on `omega`.

        $${ L_a f(U) = -i (d/d omega) f(e^{-i* omega *tau_a} @ U) |_{\\omega=0} }$$
        $${ R_a f(U) = +i (d/d omega) f(U @ e^{-i* omega *tau_a}) |_{\\omega=0} }$$

        (NOTE: this is a slower implementation compared to the one using the chain rule)
        
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
        omega_shape = (B, 2, Ng, *U.shape[1:-2],) # (B, 2, L1,...,Ld, d): one omega per configuration, Left/right, link - no color index
        omega = torch.zeros(size=omega_shape, dtype=U.real.dtype, device=U.device, requires_grad=True)

        d, M = torch.linalg.eigh(self.tau) # diagonalization of the generators tau_a
        phase = torch.einsum("BDa...,ai->BDa...i", -omega, d) # angle phases
        exp_iphase = torch.exp(1j*phase) # diagonal matrix from diagonal entries
        exp_iD = torch.diag_embed(exp_iphase)
        V = torch.einsum("aij,BDa...jk,akm->BDa...im", M, exp_iD , M.adjoint())  # V_a = M exp(iD) M^\dagger
        U_prime = torch.stack(
            [
                torch.einsum("Ba...ij,B...jk->Ba...ik", V[:,0,...], U),
                torch.einsum("B...ij,Ba...jk->Ba...ik", U, V[:,1,...])
            ],
            dim=1
        ).view(*( (B*2*Ng,) + U.shape[1:] )) # flattening non-geometric dimensions
        f_U = f(GaugeConfiguration(U_prime)) # shape: (B*Ng,1)
        df_domega = my_autograd(y=f_U, x=omega, grad_outputs=torch.ones_like(f_U.real), create_graph=True, retain_graph=True)
        imag_unit_factor = torch.tensor([-1j, +1j], device=U.device) # factor in front: `-i` or `+i`
        momenta = torch.einsum("D,bD...->bD...", imag_unit_factor, df_domega) # shape (B,2,Ng,...), where index 1 is for Left(0) or Right(1) momenta
        return momenta
    
    def apply_LaRa_with_chain_rule(
            self,
            f_U: torch.tensor,
            U: torch.tensor,
            delta: torch.tensor):
        """
        (faster implementation compared to the one using the exponentials of the generators)

        Action of Left and Right canonical momenta of a scalar function f(U), using the Chain Rule.
        It returns a tensor of shape (batchsize, 2, a, ..., Nc, Nc). "2" is for either Left or Right momentum

        Notes:
            1. The function `f` returns a shape (batchsize, 1): one function value for each configuration
            2. `df/dU=df(U+\\delta)/d\\delta |_{\\delta=0}`, with `delta` complex

        """
        # d(e^{-i*omega*tau_a} U)/domega at omega==0
        # shape: (batchsize, a, ..., Nc, Nc)
        A_L = -1j * torch.einsum("aij,B...jk->Ba...ik", self.tau, U.as_subclass(torch.Tensor))
        # d(U e^{+i*omega*tau_a})/domega at omega==0
        # shape: (batchsize, a, ..., Nc, Nc)
        A_R = -1j * torch.einsum("B...ij,ajk->Ba...ik", U.as_subclass(torch.Tensor), self.tau)
        A = torch.stack((A_L, A_R), dim=1) # (batchsize, 2, a, ..., Nc, Nc) | "2" is for either Left or Right momentum

        f_U_flat = f_U.squeeze(dim=1)
        dRef_dU = my_autograd(y=f_U_flat.real, x=delta, grad_outputs=torch.ones_like(f_U_flat.real), create_graph=False).unsqueeze(1).unsqueeze(1)
        dImf_dU = my_autograd(y=f_U_flat.imag, x=delta, grad_outputs=torch.ones_like(f_U_flat.imag), create_graph=False).unsqueeze(1).unsqueeze(1)
        df_domega = chain_rule_contributions(A=A, dRef_dU=dRef_dU, dImf_dU=dImf_dU).sum(dim=(-2,-1)) # summing over the color components

        imag_unit_factor = torch.tensor([-1j, +1j], device=U.device) # factor in front: `-i` or `+i`
        momenta = torch.einsum("D,bD...->bD...", imag_unit_factor, df_domega) # shape (B,2,Ng,...), where index 1 is for Left(0) or Right(1) momenta
        return momenta

    def LaRa_chain_rule(self, f: typing.Callable, U: GaugeConfiguration):
        """ L_a(f(U)) and R_a(f(U)) in one go"""
        #delta = torch.zeros(size=U.shape[1:], dtype=U.dtype, device=U.device, requires_grad=True) # Note: delta should be complex
        #f_U = f(GaugeConfiguration(U + delta.unsqueeze(0))) # f(U+delta)
        delta = torch.zeros(size=U.shape, dtype=U.dtype, device=U.device, requires_grad=True) # Note: delta should be complex
        f_U = f(GaugeConfiguration(U + delta)) + 0.0*1j
        assert(f_U.shape == (U.batch_size, 1)) # scalar output, one for each batch 
        #f_U_sum = f_U.sum() + 0.0*1j # f(U+delta)
        return self.apply_LaRa_with_chain_rule(f_U=f_U, U=U, delta=delta)
    #---

    def La_chain_rule(self, f: typing.Callable, U: GaugeConfiguration):
        """
        batched Left canonical momenta using the chain rule:

        $$
        -i (d/d omega) f(e^{-i* omega *tau_a} @ U) |_{omega=0} =
        -i * (-i*\\tau_a U)_{ab} (\\partial f / \\partial U_{ab})
        $$

        Note: `df/dU=df(U+\\delta)/d\\delta |_{\\delta=0}`
        """
        delta = torch.zeros(size=U.shape, dtype=U.dtype, device=U.device, requires_grad=True) # Note: delta should be complex
        f_U = f(GaugeConfiguration(U + delta)) # f(U+delta)
        # NOTE: I can differentiate the sum over configurations
        # because f(U) acts configuration-wise
        f_U_flat = f_U.sum() + 0.0*1j # flattened view
        A = -1j * torch.einsum("aij,B...jk->Ba...ik", self.tau, U.as_subclass(torch.Tensor)) # d(e^{-i*omega*tau_a})/domega at omega==0
        dRef_dU = my_autograd(y=f_U.real, x=delta, grad_outputs=torch.ones_like(f_U.real), create_graph=False).unsqueeze(1)
        dImf_dU = my_autograd(y=f_U.imag, x=delta, grad_outputs=torch.ones_like(f_U.imag), create_graph=False).unsqueeze(1)
        df_domega = chain_rule_contributions(A=A, dRef_dU=dRef_dU, dImf_dU=dImf_dU).sum(dim=(-2,-1)) # summing over the color components
        La_f = -1j * df_domega
        return La_f
