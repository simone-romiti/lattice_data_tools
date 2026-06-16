"""Implementation of a class representing a gauge configuration"""

from abc import abstractmethod
import numpy as np
import torch
import typing


import lattice_data_tools.links.suN as suN
from lattice_data_tools.links.loops import WilsonLoopsGenerator


class ColorMatrix(torch.Tensor):
    """
    Tensor representing a set of color matrices:
    (..., Nc, Nc)
    """
    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     instance = torch.Tensor._make_subclass(cls, x)
    #     return instance
    
    def __init__(self, x, *args, **kwargs):
        self._data = x
        self._Nc = x.shape[-1]  # read from x, not instance

    def validate(self):
        """Check that it is actually a set of color matrice of size Nc \\times Nc"""
        N1, N2 = self.shape[-2:]
        if N1 != N2:
            raise ValueError(f"Invalid shape {self.shape}, expected {expected}")
    #-------

    def dim(self):
        raise AttributeError(
            "'ColorMatrix' intentionally disables .dim(). "
            "Use .n_dims for the number of lattice dimensions, "
            "or .as_subclass(torch.Tensor).dim() for the tensor rank."
        )

    def __mul__(self, other):
        raise TypeError("'ColorMatrix' disables '*'. Use '@' for matrix multiplication.")

    def __rmul__(self, other):
        raise TypeError("'ColorMatrix' disables '*'. Use '@' for matrix multiplication.")

    def __matmul__(self, other):
        A = self.as_subclass(torch.Tensor)
        B = other.as_subclass(torch.Tensor) if isinstance(other, torch.Tensor) else other
        return torch.matmul(A,B)

    def __rmatmul__(self, other):
        A = other.as_subclass(torch.Tensor) if isinstance(other, torch.Tensor) else other
        B = self.as_subclass(torch.Tensor)
        return torch.matmul(A,B) 
                            
    #---

    def __repr__(self):
        return self.as_subclass(torch.Tensor).__repr__()
    def __str__(self):
        return self.as_subclass(torch.Tensor).__str__()

    def to_tensor(self):
        return self.as_subclass(torch.Tensor)

    @property
    def Nc(self):
        return self._Nc

    @property
    def Ng(self) -> int:
        """Number of generators of the Lie algebra: Nc^2 - 1 for SU(Nc)."""
        return suN.get_Ng(Nc=self.Nc)

    def dagger(self):
        return self.adjoint()

    def get_traceless_part(self):
        """ Returns the traceless version of the object """
        Nc = self.Nc
        # Compute trace over last two dims: shape (...)
        tr = torch.diagonal(self, dim1=-1, dim2=-2).sum(-1).to_tensor()
        # Subtract (tr/Nc) * I, broadcasting over batch dims
        eye = torch.eye(Nc, dtype=self.dtype, device=self.device)
        res = self - (tr / Nc)[..., None, None] * eye
        return res
    #---
    def get_ah_traceless(self):
        """
        Computes the anti-Hermitian traceless part of W:
            W_aht = (W - W†)/2 - (Tr((W - W†)/2) / Nc) * I
        Avoids intermediate rounding by combining steps.
        """
        Nc = self.Nc
        eye = torch.eye(Nc, dtype=self.dtype, device=self.device)

        # Anti-Hermitian part
        W_ah = (self - self.adjoint()) / 2.0

        # Trace is purely imaginary; take only imaginary part of diagonal to avoid rounding
        tr = torch.diagonal(W_ah.to_tensor(), dim1=-1, dim2=-2).sum(-1)
        tr = 1j * tr.imag  # force exact purely-imaginary trace

        return W_ah - (tr/Nc)[..., None, None] * eye   

    

class GaugeConfiguration(ColorMatrix):
    """
    Tensor representing a gauge configuration of links (in the fundamental representation) with shape:
    (B, L1, ..., Ld, d, Nc, Nc)
    """
    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     instance = torch.Tensor._make_subclass(cls, x)
    #     return instance
    
    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        self._batch_size    = x.shape[0]
        self._lattice_shape = x.shape[1:-3]
        self._d             = x.shape[-3]
        self.validate()
    
    def validate(self):
        """Check that shape == (B, L1, ..., Ld, d, Nc, Nc)"""
        expected = (self.batch_size, *self.lattice_shape, self.n_dims, self.Nc, self.Nc)
        nL = len(self.lattice_shape)
        if nL != self.n_dims:
            raise ValueError(f"Number of (L1,...,Ld)={nL}. This is different from d={self.n_dims}")
        if tuple(self.shape) != expected:
            raise ValueError(f"Invalid shape {self.shape}, expected {expected}")
    #-------
    def save(self, path: str):
        """ saving batch of configurations (e.g. to `.pt` file )"""
        torch.save(self.as_subclass(torch.Tensor), path)

    @staticmethod
    def load(path: str):
        """ loading configuration (e.g. from a `.pt` file)"""
        return GaugeConfiguration(torch.load(f=path))

    @property
    def batch_size(self):
        return self._batch_size
    @property
    def batchsize(self):
        return self.batch_size

    @property
    def lattice_shape(self):
        return self._lattice_shape

    @property
    def n_dims(self):
        return self._d

    @property
    def n_links(self):
        return (torch.prod(torch.tensor(self.lattice_shape)) * self.n_dims).item()

    @property
    def Nc(self):
        return self._Nc

    @property
    def Ng(self) -> int:
        """Number of generators of the Lie algebra: Nc^2 - 1 for SU(Nc)."""
        return suN.get_Ng(Nc=self.Nc)

    @staticmethod
    def from_theta(theta: torch.Tensor) -> "GaugeConfiguration":
        """
        Build U_\\mu(x) = exp(i * \\theta^a_\\mu(x) * tau_a)
        theta: shape (B, L1, ..., Ld, d, Ng)

        NOTE: inverse function does not exist --> see implementation
        """
        return GaugeConfiguration(suN.get_U_from_theta(theta=theta))
    #---
    @staticmethod
    def from_hotstart(batchsize: int, L_mu: typing.List[int], Nc: int, seed: int,  dtype: torch.dtype, device: torch.device, requires_grad: bool):
        """
        hot configuration of U_\\mu(x) (all identities in SU(Nc)) from the dimensions

        output shape = (batch_size, L1,...,Ld, d, Nc, Nc)
        """
        d = len(L_mu)
        shape = (batchsize, *L_mu, d, Nc, Nc)
        U_tensor = suN.get_hotstart(shape=shape, seed=seed, dtype=dtype, device=device, requires_grad=False)
        U =  GaugeConfiguration(U_tensor)
        U.requires_grad_(requires_grad=requires_grad) # start tracking grads from here
        return U

    def hotstart(self, seed: int) -> None:
        suN.apply_hotstart(U=self, seed=seed)
        return None
    #---
    def coldstart(self) -> None:
        suN.apply_coldstart(U=self)
        return None
    #---

    @staticmethod
    def from_coldstart(batchsize: int, L_mu: typing.List[int], Nc: int, dtype: torch.dtype, device: torch.device, requires_grad: bool):
        """
        cold configuration of U_\\mu(x) (all identities in SU(Nc)) from the dimensions

        output shape = (batch_size, L1,...,Ld, d, Nc, Nc)
        """
        d = len(L_mu)
        shape = (batchsize, *L_mu, d, Nc, Nc)
        return GaugeConfiguration(suN.get_coldstart(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad))
    
    def Left_gauge_transformation(self, V: torch.Tensor) -> None:
        """
        Apply a LEFT gauge transformation to this object:

        U_\\mu(x) \\to V(x) @ U_\\mu(x)

        V: array fo shape (batchsize, L1,...,Ld, Nc, Nc)
        """
        self.copy_(V.unsqueeze(-3).expand(*(self.shape)) @ self)
        return None
    #---
    def Right_gauge_transformation(self, V: torch.Tensor) -> None:
        """
        Apply a LEFT gauge transformation to this object:

        U_\\mu(x) \\to U_\\mu(x) @ V(x + \\mu)^\\dagger

        V: array fo shape (batchsize, L1,...,Ld, Nc, Nc)
        """
        for mu in range(self.n_dims):
            V_xpmu = torch.roll(input=V, shifts=-1, dims=1+mu) # V(x + \\mu)^\\dagger
            self[...,mu,:,:] = (self[...,mu,:,:] @ V_xpmu.adjoint())
        #---
        return None
    #---
    def gauge_transformation(self, V: torch.Tensor) -> None:
        """ Transforming the configuarion of links """
        self.Left_gauge_transformation(V=V)
        self.Right_gauge_transformation(V=V)
        return None
    #---
    def gen_random_gauge_transformation(self, seed: int) -> None:
        """
        Returns a random SU(Nc) tensor representing the matrices V(x) of a gauge tranformation,
        with the correct shape for the present configuration
        """
        V = self[...,0,:,:].clone() # same shape of U
        suN.apply_hotstart(V, seed=seed) # setting V to be a random transformation
        return V
    #---
    def random_gauge_transformation(self, seed: int) -> None:
        self.gauge_transformation(V=self.get_random_gauge_transformation(seed=seed))
        return None
    #---

    def plaquette(self, x: torch.tensor, mu, nu):
        x_pmu, x_pnu = x.clone(), x.clone()
        x_pmu[mu] += 1
        x_pnu[nu] += 1
        U1 = self[:,*x,mu,:,:] # U_\\mu(x)
        U2 = self[:,*x_pmu,nu,:,:] # U_\\nu(x+mu)
        U3 = self[:,*x_pnu,mu,:,:].adjoint() # U_\\mu(x+nu)^\\dagger
        U4 = self[:,*x,nu,:,:].adjoint() # U_\nu(x)^\\dagger
        P_munu = U1 @ U2 @ U3 @ U4
        return P_munu
    
    def plaquettes(self):
        """ plaquettes: shape = (B, L1, ..., Ld, n_plaq, Nc, Nc)"""
        P = WilsonLoopsGenerator.plaquettes(U=self)
        return P
    #---
    def average_Tr_plaquette(self):
        """ average plaquette for each configuration: shape=(B,) """
        P = self.plaquettes()
        trP = suN.get_Tr(P).as_subclass(torch.Tensor)
        idx_sum = tuple(torch.arange(1, len(trP.shape)))
        avg_trP = trP.mean(dim=idx_sum)
        return avg_trP

    def average_ReTr_plaquette(self):
        return self.average_Tr_plaquette().real

    def Wilson_action(self, beta):
        """
        Wilson action as -(beta/Nc) \\sum_{\\box} U_{\\box}
        """
        P = self.plaquettes()
        trP = suN.get_Tr(P).as_subclass(torch.Tensor)
        idx_sum = tuple(torch.arange(1, len(trP.shape)))
        sum_trP = trP.sum(dim=idx_sum)
        sum_ReTr = sum_trP.real
        S = -(beta/self.Nc) * sum_ReTr
        return S
#---

class LocallyGaugeCovariant(ColorMatrix):
    """
    Tensor representing a configuration of N_obs gauge covariant variables that transform locally under a gauge transformation:

    W_i(x) --> V(x) W_i(x) V^{\\dagger}(x)

    shape: (B, L1, ..., Ld, N_obs, Nc, Nc)
    """
    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     instance = super().__new__(cls, x, *args, **kwargs)
    #     instance._batch_size = instance.shape[0]
    #     instance._lattice_shape = instance.shape[1:-3]
    #     instance._N_obs = instance.shape[-3]
    #     instance.validate()
    #     return instance

    def __init__(self, x, *args, **kwargs):
        super().__init__(x, *args, **kwargs)
        self._batch_size = self.shape[0]
        self._lattice_shape = self.shape[1:-3]
        self._N_obs = self.shape[-3]
        self.validate()

    def validate(self):
        """Check that shape == (B, L1, ..., Ld, d, Nc, Nc)"""
        expected = (self.batch_size, *self.lattice_shape, self.N_obs, self.Nc, self.Nc)
        if tuple(self.shape) != expected:
            raise ValueError(f"Invalid shape {self.shape}, expected {expected}")
    #-------
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def lattice_shape(self):
        return self._lattice_shape

    @property
    def N_obs(self):
        return self._N_obs

    def gauge_transformation(self, V: torch.Tensor) -> None:
        """
        Local gauge transformation:

        U_\\mu(x) \\to V(x) U_\\mu(x) @ V(x)^\\dagger
        
        """
        self.copy_(V[...,None,:,:] @ self @ V[...,None,:,:].adjoint()) #  V(x) U_\\mu(x) V(x)^\\dagger
        return None
    #---
    def random_gauge_transformation(self, seed: int) -> None:
        V = self[...,0,:,:].clone() # same shape of U
        suN.apply_hotstart(V, seed=seed) # setting V to be a random transformation
        self.gauge_transformation(V=V)
        return None
    #---
    @staticmethod
    def check_gauge_covariance(U: GaugeConfiguration, V: torch.Tensor, fun: typing.Callable[["GaugeConfiguration"], "LocallyGaugeCovariant"], atol : float = 1e-15):
        """
        Checks that W=fun(U) transforms locally:

        W(x) --> V(x) W(x) V(x)^\\dagger

        This is done computing the object transformed according to what we expect (the local transformation),
        and verifying that it is the same as the object computed on the transformed gauge configuration.
        In other words, we check that the function for W(U) is well implemented, as the order of gauge transformations is 
        
        """
        U1 = GaugeConfiguration(U.clone()) # copy of the original configuration
        W1 = fun(U1) # function of the original configuration of links
        U1.gauge_transformation(V=V) # inplace gauge transformation of the links
        W2 = fun(U1) # W(U) computed on the new configuration
        assert (not torch.allclose(W1, W2, atol=atol)), "W(U) should change. Is your lambda function W(U) well defined?" # check that W1 and W2 are different (because we have not transformed W1 yet)
        W1.gauge_transformation(V=V) # transforming the object W1
        assert(torch.allclose(W1, W2, atol=atol)) # check that the order of gauge transformations does matter
    #---
#---


