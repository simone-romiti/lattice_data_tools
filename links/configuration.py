"""Implementation of a class representing a gauge configuration"""

from abc import abstractmethod
import numpy as np
import torch
import sys
sys.path.append("../../")
import lattice_data_tools.links.suN as suN


class ColorMatrix(torch.Tensor):
    """
    Tensor representing a configuration of color matrices:
    (B, L1, ..., Ld, d, ..., Nc, Nc)
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        instance = torch.Tensor._make_subclass(cls, x)
        instance._lattice_shape = instance.shape[1:-3]
        instance._d = instance.shape[1 + len(instance._lattice_shape)]
        instance._Nc = instance.shape[-1]
        return instance

    def __init__(self, x, *args, **kwargs):
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, ColorMatrix):
            src = next((a for a in torch.utils._pytree.tree_leaves(args)
                        if isinstance(a, ColorMatrix)), None)
            if src is not None:
                ret._lattice_shape = src._lattice_shape
                ret._d = src._d
                ret._Nc = src._Nc
        return ret

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
    def batch_size(self):
        return self.shape[0]

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

    def dagger(self):
        return self.adjoint()

    @abstractmethod
    def validate(self):
        pass


class GaugeConfiguration(ColorMatrix):
    """
    Tensor representing a gauge configuration (in the fundamental representation) with shape:
    (B, L1, ..., Ld, d, Nc, Nc)
    """
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        instance = super().__new__(cls, x, *args, **kwargs)
        instance.validate()
        return instance

    def __init__(self, x, *args, **kwargs):
        pass

    def validate(self):
        """Check that shape == (B, L1, ..., Ld, d, Nc, Nc)"""
        expected = (self.batch_size, *self.lattice_shape, self.n_dims, self.Nc, self.Nc)
        if tuple(self.shape) != expected:
            raise ValueError(f"Invalid shape {self.shape}, expected {expected}")

    @property
    def Ng(self):
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
        suN.apply_hotstart(seed=seed)
        return None

    def Left_gauge_transformation(self, V: torch.Tensor):
        """
        Apply a LEFT gauge transformation to this object:

        U_\\mu(x) \\to V(x) @ U_\\mu(x)

        V: array fo shape (batchsize, L1,...,Ld, Nc, Nc)
        """
        self = V.unqueeze(-1).expand(*(self.shape)) @ self
    #---
    def Right_gauge_transformation(self, V: torch.Tensor):
        """
        Apply a LEFT gauge transformation to this object:

        U_\\mu(x) \\to U_\\mu(x) @ V(x + \\mu)^\\dagger

        V: array fo shape (batchsize, L1,...,Ld, Nc, Nc)
        """
        for mu in range(self.n_dims):
            V_xpmu = torch.roll(torch.roll(V, -1, dims=1+nu)) # V(x + \\mu)^\\dagger
            self = self @ V_xpmu.adjoint()
    #---
    def gauge_transformation(V: torch.Tensor):
        self.Left_gauge_transformation(V=V)
        self.Right_gauge_transformation(V=V)
    #---
        

if __name__ == "__main__":
    device = torch.device("cpu")
    B = 1
    d = 4
    Lmu = d * [12]
    Nc = 3
    Ng = suN.get_Ng(Nc=Nc)
    # random angles in [-\pi, \pi]
    theta = -torch.pi + (2 * torch.pi) * torch.rand(B, *Lmu, d, Ng).type(torch.float64)
    U = GaugeConfiguration.from_theta(theta)
    U.hotstart(seed=12345)
    Udag = U.adjoint()
    print("Type and shape of the gauge configuration")
    print("U:", type(U), U.shape)
    print("Udag:", type(Udag), U.shape)
    print("Unitarity check:", torch.allclose(U @ Udag, torch.eye(Nc).type(U.type())))
    print("B=", U.batch_size)
    print("Lattice shape: ", U.lattice_shape)
    print("d=", U.n_dims)
    print("n_links=", U.n_links)
    print("Nc=", U.Nc)
    # behaviour checks
    print("\n behaviour checks \n")
    print("U + U type:", type(U+U))
    print("U - U type:", type(U-U))
    print("U @ Udag type:", type(U @ Udag))
    try:
        _ = U * U
    except TypeError as e:
        print(f"U * U correctly raised TypeError: {e}")
    try:
        _ = U.dim()
    except AttributeError as e:
        print(f"U.dim() correctly raised AttributeError: {e}")
    #---
    print("Applying gauge transformations")
    V = suN.apply_hotstart(U[...,0,:,:].clone())
    U.gauge_transformation(V=V)
    
    
