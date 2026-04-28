"""Implementation of a class representing a gauge configuration"""

from abc import abstractmethod
import numpy as np
import torch
import sys
sys.path.append("../../")
from lattice_data_tools.links.suN import get_Ng, get_U_from_theta


class color_matrices(torch.Tensor):
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
        if isinstance(ret, color_matrices):
            src = next((a for a in torch.utils._pytree.tree_leaves(args)
                        if isinstance(a, color_matrices)), None)
            if src is not None:
                ret._lattice_shape = src._lattice_shape
                ret._d = src._d
                ret._Nc = src._Nc
        return ret

    def dim(self):
        raise AttributeError(
            "'color_matrices' intentionally disables .dim(). "
            "Use .n_dims for the number of lattice dimensions, "
            "or .as_subclass(torch.Tensor).dim() for the tensor rank."
        )

    def __mul__(self, other):
        raise TypeError("'color_matrices' disables '*'. Use '@' for matrix multiplication.")

    def __rmul__(self, other):
        raise TypeError("'color_matrices' disables '*'. Use '@' for matrix multiplication.")

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


class GaugeConfiguration(color_matrices):
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
        return get_Ng(Nc=self.Nc)

    @staticmethod
    def from_theta(theta: torch.Tensor) -> "GaugeConfiguration":
        """
        Build U_\\mu(x) = exp(i * \\theta^a_\\mu(x) * tau_a)
        theta: shape (B, L1, ..., Ld, d, Ng)

        NOTE: inverse function does not exist --> see implementation
        """
        return GaugeConfiguration(get_U_from_theta(theta=theta))
    #---

    def hotstart(self, seed: int):
        """
        Initialize gauge links to random SU(Nc) elements,
        following the recipe of page 11 of https://arxiv.org/pdf/math-ph/0609050

        Each link matrix is constructed by:
          1. Drawing an Nc×Nc random complex matrix.
          2. Applying row-wise Gram-Schmidt orthonormalization → U(Nc) matrix.
          3. Dividing the first row by the Nc-th root of det(U) → SU(Nc) matrix.

        Returns: GaugeConfiguration  shape: (B, L1, ..., Ld, d, Nc, Nc)
        """
        torch.manual_seed(seed) # seeting the RNG seed for reproducibility
        # generating a random complex matrix
        imag_part = torch.randn(self.shape, dtype=self.real.dtype, device=self.device) # real part
        real_part = torch.randn(self.shape, dtype=self.imag.dtype, device=self.device) # imaginary part
        Z = (real_part + 1j*imag_part)/np.sqrt(2.0)
        Q, R = torch.linalg.qr(Z) # QR decomposition: https://en.wikipedia.org/wiki/QR_decomposition
        diag = torch.diagonal(R, dim1=-2, dim2=-1) # (..., Nc) --> extracts R_ii
        signs = diag / diag.abs() # R_ii / |R_ii|
        Lam = torch.diag_embed(signs) # shape: (..., Nc, Nc). Eq. 5.12 of https://arxiv.org/pdf/math-ph/0609050
        Qprime = Q @ Lam
        detQprime = torch.linalg.det(Qprime).unsqueeze(-1).unsqueeze(-1) # det(Q). reshaping in order to combine with Q later
        U = Q / detQprime # Q is unitary, we need to impose det(U)==1
        self = GaugeConfiguration(U)
        return None

    def gauge_transformation(self, V):
        """
        Apply a gauge transformation to this object:

        U_\\mu(x) \\to V(x) @ U_\\mu(x) @ V^\\dagger(x+\\mu)
        """
    

if __name__ == "__main__":
    device = torch.device("cpu")
    B = 1
    d = 4
    Lmu = d * [12]
    Nc = 3
    Ng = get_Ng(Nc=Nc)
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
