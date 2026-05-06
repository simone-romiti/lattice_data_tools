"""Implementation of a class representing a gauge configuration"""

from abc import abstractmethod
import numpy as np
import torch
import typing


import lattice_data_tools.links.suN as suN


class ColorMatrix(torch.Tensor):
    """
    Tensor representing a set of color matrices:
    (..., Nc, Nc)
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        instance = torch.Tensor._make_subclass(cls, x)
        instance._Nc = instance.shape[-1]
        return instance

    def __init__(self, x, *args, **kwargs):
        pass

    def validate(self):
        """Check that it is actually a set of color matrice of size Nc \\times Nc"""
        N1, N2 = self.shape[-2:]
        if N1 != N2:
            raise ValueError(f"Invalid shape {self.shape}, expected {expected}")
    #-------
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, ColorMatrix):
            src = next((a for a in torch.utils._pytree.tree_leaves(args)
                        if isinstance(a, ColorMatrix)), None)
            if src is not None:
                ret._Nc = src._Nc
        return ret
    #---

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

    _DOWNCAST_OPS = {
        torch.Tensor.__add__,
        torch.Tensor.__radd__,
        torch.Tensor.__sub__,
        torch.Tensor.__rsub__,
        torch.add,
        torch.sub,
    }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if func in cls._DOWNCAST_OPS:
            if isinstance(ret, torch.Tensor):
                return ret.as_subclass(torch.Tensor)
        if isinstance(ret, GaugeConfiguration):
            try:
                ret.validate()
            except (ValueError, AttributeError):
                return ret.as_subclass(torch.Tensor)
        return ret

    @property
    def Nc(self):
        return self._Nc

    @property
    def Ng(self) -> int:
        """Number of generators of the Lie algebra: Nc^2 - 1 for SU(Nc)."""
        return suN.get_Ng(Nc=self.Nc)

    def dagger(self):
        return self.adjoint()

    def to_tensor(self):
        return torch.Tensor(self)
    
    @abstractmethod
    def validate(self):
        pass


class GaugeConfiguration(ColorMatrix):
    """
    Tensor representing a gauge configuration of links (in the fundamental representation) with shape:
    (B, L1, ..., Ld, d, Nc, Nc)
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        instance = super().__new__(cls, x, *args, **kwargs)
        instance._batch_size = instance.shape[0]
        instance._lattice_shape = instance.shape[1:-3]
        instance._d = instance.shape[1 + len(instance._lattice_shape)]
        instance._Nc = instance.shape[-1]
        instance.validate()
        return instance

    def __init__(self, x, *args, **kwargs):
        pass

    def validate(self):
        """Check that shape == (B, L1, ..., Ld, d, Nc, Nc)"""
        expected = (self.batch_size, *self.lattice_shape, self.n_dims, self.Nc, self.Nc)
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
    def hotstart(self, seed: int) -> None:
        suN.apply_hotstart(U=self, seed=seed)
        return None
    #---
    def coldstart(self, seed: int) -> None:
        suN.apply_coldstart(U=self, seed=seed)
        return None
    #---
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
        print("## Transforming a conf. of links")
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
#---

class LocallyGaugeCovariant(ColorMatrix):
    """
    Tensor representing a configuration of N_obs gauge covariant variables that transform locally under a gauge transformation:

    W_i(x) --> V(x) W_i(x) V^{\\dagger}(x)

    shape: (B, L1, ..., Ld, N_obs, Nc, Nc)
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        instance = super().__new__(cls, x, *args, **kwargs)
        instance._batch_size = instance.shape[0]
        instance._lattice_shape = instance.shape[1:-3]
        instance._N_obs = instance.shape[-3]
        instance.validate()
        return instance

    def __init__(self, x, *args, **kwargs):
        pass

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
        print("## Transforming a local object")
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
        
        
